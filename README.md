# oai-to-gemini

一个给 [New API](https://github.com/Calcium-Ion/new-api) 使用的中转服务：

- 客户端（New API）按 OpenAI 兼容格式请求本服务。
- 本服务按路径中的供应商名选择上游，把请求转换成 Gemini 兼容格式。
- 上游返回 Gemini 格式后，本服务再转回 OpenAI 兼容格式。

核心约束：**任何文本块拼接都原样拼接，绝不插入 `\n`、空格或其他分隔符。** 单元测试和代理集成测试都覆盖了这一点。

## 功能概览

- 多供应商配置，按 `/<供应商名>/...` 路由；未知供应商返回 OpenAI 格式 404，不会回退到默认供应商。
- `POST /<供应商>/v1/chat/completions` 和 `POST /<供应商>/chat/completions`。
- `GET /<供应商>/v1/models` 和 `GET /<供应商>/models`（Gemini 模型列表会转成 OpenAI 列表并合并全部分页；已是 OpenAI 格式则原样透传）。上游模型列表会校验 OpenAI `data` 或 Gemini `models` 的结构，并逐条要求非空模型标识：OpenAI 条目必须有非空字符串 `id`，Gemini 条目必须有非空且可转换的 `name`。单个模型标识最长 8 KiB，跨分页累计模型名字节数上限 2 MiB；单个分页令牌最长 8 KiB，累计令牌字节上限 1 MiB，循环检测只保存令牌 SHA-256 摘要，下一页只保留当前实际令牌。HTTP 200 内嵌错误、无法识别的成功结构、缺字段、空标识、超长标识、累计字节超限或有效/无效条目混合都返回 502；合法空列表仍返回 `data: []`。
- `GET /healthz` 返回 `ok`。
- 非流式、流式（SSE）、以及非流式请求通过上游 SSE 聚合的 `aggregate_sse` 模式。
- SSE 事件按行增量收集，超过阈值后连同文本和图片载荷一起溢出到随机命名的临时文件，完成即删除；`sse_cache_max_bytes` 是已完成缓存文件的软上限，不会截断正在聚合的响应。SSE 解析阶段会对候选、part、未知字段和非载荷元数据实施数量与累计字节预算；聚合保留的工具调用、图片 MIME 和偏移等元数据另有数量与字节预算。
- 原样转发客户端的 `Authorization`、`x-goog-api-key` 和 `?key=`；不读取、不保存、不打印密钥。
- Gemini 返回的图片 `inlineData` 会转成 OpenAI `image_url` content part；其他暂不支持或未建模的响应内容类型返回明确错误，不会静默丢弃。非流式、流式和聚合路径统一使用 `unsupported_response` 错误码。
- 上游 TCP 建连、请求发送、响应体、下游请求读取和下游响应写出都有超时限制；流式路径的上游空闲超时只在等待上游数据时计时，下游写出使用独立的写空闲超时，不会用固定总时长或下游背压误伤持续正常传输的长响应。
- 收到 `SIGINT`/`SIGTERM` 后先停止接收新请求，最多等待 10 秒让活动请求自然完成；超时则取消活动请求并关闭剩余连接，等 handler 释放聚合缓冲后再删除本实例缓存目录。正在聚合、已落盘的请求不会因为退出而提前失去缓存文件。

## 快速开始（本地）

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，把 example.invalid 换成你的上游地址（不要写真实密钥）
go run . -config config.yaml
```

默认监听 `:8080`，也可以用 `-listen :8080` 或配置里的 `listen` 覆盖。配置文件默认查找顺序：

1. `-config` 参数
2. 环境变量 `CONFIG_PATH`
3. 当前目录 `config.yaml`
4. 当前目录 `config.json`

检查健康状态：

```bash
curl -s http://127.0.0.1:8080/healthz
# ok
```

发起一次非流式请求（token 用占位符，实际由 New API/客户端传入）：

```bash
curl -s http://127.0.0.1:8080/newapi1/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <upstream-token>' \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

## 配置

`config.example.yaml`：

```yaml
listen: ":8080"
sse_spill_threshold_bytes: 1048576
sse_cache_max_bytes: 67108864
# sse_cache_dir: /tmp/oai-to-gemini-sse

providers:
  - name: newapi1
    api_base: https://example.invalid/v1beta
  - name: newapi2
    base_url: https://example.invalid/v1beta
```

- `name` 必须叫 `name`，且必须唯一；它就是请求路径的第一段。名称不能包含空白、`/`、`?`、`#`
  或 `%`。请求路径由 Go HTTP 层统一百分号解码一次，路由不再二次解码，因此不会出现 `%61` 被
  误解码成 `a` 并把凭证转发到另一个供应商的问题。
- `api_base` 是规范字段，也兼容 `api-base` 和 `base_url` 两个别名。
- `api_base` 应指向 Gemini API 的版本根，例如 `https://<host>/v1beta`。本服务会拼成
  `.../models/<model>:generateContent` 或 `.../models/<model>:streamGenerateContent?alt=sse`。
  版本根不能包含 query 或 fragment；包含时配置加载会直接报错。
- `sse_spill_threshold_bytes`：聚合时内存阈值，默认 1 MiB。
- `sse_cache_max_bytes`：每个实例已完成/遗留 SSE 临时文件的软上限，默认 64 MiB。**不是**单次响应
  截断上限，也不限制正在聚合的响应。
- `sse_cache_dir`：可选，缓存基目录；不填则使用系统临时目录下的 `oai-to-gemini-sse`。每个进程
  在基目录下创建自己的 `run-*` 实例子目录并加目录锁；多个实例可以安全共用同一个基目录，清理器
  不会删除仍被其他实例锁定使用的目录。启动时会清理超过宽限期、且未被锁定的旧实例目录，以及
  旧版本在系统临时目录留下的 `oai-to-gemini-sse-*` 目录（只处理本服务格式的旧溢出文件）。
- 配置里不存放任何密钥。

也支持 `config.json`，字段名相同，参见 `config.example.json`。

## 路由规则

| 请求 | 行为 |
| --- | --- |
| `POST /<name>/v1/chat/completions` | 转发到该供应商的 Gemini `generateContent` 或 `streamGenerateContent` |
| `POST /<name>/chat/completions` | 同上，兼容不带 `/v1` 的路径 |
| `GET /<name>/v1/models` | 转发到上游 `/models`，Gemini 列表转换为 OpenAI 列表 |
| `GET /<name>/models` | 同上 |
| `GET /healthz` | 返回 `ok` |
| `/<unknown>/...` | 404，OpenAI 错误格式，不请求任何上游 |

转发时去掉开头的 `/<name>`，剩余路径拼到 `api_base` 后面。例如
`api_base=https://example.invalid/v1beta` 时，`/newapi1/v1/chat/completions` 会请求
`https://example.invalid/v1beta/models/<model>:generateContent`。

## OpenAI -> Gemini 转换

- `model` 原样用作 Gemini 模型名（应传裸模型名，例如 `gemini-2.0-flash`）。
- `system` / `developer` 消息合并为 `systemInstruction.parts[].text`；多条 system 消息直接拼接，不加分隔符。
- `user` -> `role: user`，`assistant` -> `role: model`。
- `content` 为字符串时直接使用；为数组时支持 `text` / `input_text`、`image_url`、`input_audio`、`file`。
  相邻文本部分原样拼接，不加换行；`image_url` 的 base64 data URL 和 `input_audio.data` 转为
  `inlineData`，HTTP(S) 图片/文件 URL 转为 `fileData`。未支持的 part 类型会返回明确错误，不会静默删除。
- `tool` 消息 -> `role: user` 的 `functionResponse`。连续的多条 `tool` 消息（同一轮并行工具调用
  的结果）会合并到同一个 Gemini user 回合的 `parts` 数组里，使 `functionResponse` 数量与前面的
  `functionCall` 数量一致；不同轮次之间仍使用独立的 user 回合。会先用前面的 assistant
  `tool_calls` 建立 `tool_call_id -> function name` 映射；找不到时依次回退到消息的 `name`、`"tool"`。
  `content` 是 JSON 对象时直接作为 response；是字符串时尝试解析其中的 JSON，否则包装为
  `{"result": "..."}`；是文本 content part 数组时，先按顺序原样拼接所有文本（不加任何分隔符），
  再按字符串规则解析/包装；数组中含非文本块时返回明确错误。其他 JSON 值包装为
  `{"result": ...}`。
- `tools[].type == "function"` -> `tools[].functionDeclarations[]`。
- `tool_choice`：`auto` -> `AUTO`，`none` -> `NONE`，`required` -> `ANY`，
  指定函数对象 -> `ANY` + `allowedFunctionNames`。
- `assistant.tool_calls` -> `functionCall` 部分；`arguments` 必须是合法 JSON，非法时回退为 `{}`。
- `generationConfig` 映射：
  - `temperature` -> `temperature`
  - `top_p` -> `topP`
  - `max_tokens` / `max_completion_tokens` -> `maxOutputTokens`
  - `stop`（字符串或字符串数组）-> `stopSequences`
  - `presence_penalty` -> `presencePenalty`
  - `frequency_penalty` -> `frequencyPenalty`
  - `seed` -> `seed`
  - `response_format.type == "json_object"` -> `responseMimeType: application/json`
  - `response_format.type == "json_schema"` -> `responseMimeType: application/json` + `responseSchema`
- Gemini 风格字段尽量透传：顶层 `topK`、`candidateCount`、`presencePenalty`、`frequencyPenalty`、
  `seed`、`responseMimeType`、`responseSchema`、`responseLogprobs`、`thinkingConfig`、
  `mediaResolution`，以及嵌套的 `generationConfig` 对象。由于响应侧只支持单候选，
  `candidateCount > 1` 以及 OpenAI 的 `n > 1` 都会在请求上游前明确拒绝；`n: 1` 按单候选正常处理。
- `safetySettings` / `safety_settings` 原样作为 Gemini 顶层字段透传。
- 不会给用户没传的参数补默认值。

## Gemini -> OpenAI 转换

- 非流式：`candidates[0].content.parts` 中所有非 thought 的 `text` 原样拼接为
  `choices[0].message.content`，不加换行。
- Gemini 图片 `inlineData`（`image/*`）会转成 OpenAI 兼容的 `image_url` content part，URL 为
  `data:<mimeType>;base64,<data>`。纯文本响应仍输出字符串 `content`；文本与图片混合时输出按顺序
  排列的 content part 数组。音频、文件、`fileData`、`functionResponse` 等暂不支持的响应内容类型
  会返回明确的 `unsupported_response` 错误，不会返回空内容并伪装成成功；非流式、流式和聚合
  三条路径使用同一个错误码。未建模的 part 字段（如 `executableCode`、`codeExecutionResult`）会被
  解码层识别并返回明确错误，不会被当成空 part 跳过。
- 完整响应没有候选时，只有非空 `promptFeedback.blockReason` 被视为合法拦截；候选既无内容也无
  `finishReason` 同样视为无效。其他情况（例如 `{}`、`null`、`{"candidates":[]}`）返回
  `invalid_upstream_response` 错误，不会伪装成正常空回答。
- SSE 路径同样校验响应结构，结束标记和结构有效性分开判断：`[DONE]` 只表示流结束，不能单独
  作为成功依据；结束前必须至少收到一个有效候选（有内容或有 `finishReason`）或明确的 prompt
  block reason。空内容但带 `finishReason` 的终止候选、以及只带 `usageMetadata` 的尾事件是合法
  的；`{}`、`null`、空候选、仅 `[DONE]` 在流式和聚合路径都会返回 `invalid_upstream_response`。
- `functionCall` -> `tool_calls`，`args` 原样作为 `arguments` 字符串。
- `finishReason`：`STOP`、空值、`FINISH_REASON_UNSPECIFIED` -> `stop`，`MAX_TOKENS` -> `length`；
  `SAFETY`、`RECITATION`、`BLOCKLIST`、`PROHIBITED_CONTENT`、`SPII`、`IMAGE_SAFETY`、
  `IMAGE_PROHIBITED_CONTENT`、`IMAGE_RECITATION` -> `content_filter`。长度限制和安全拦截优先于
  工具调用：只有正常完成且实际输出了工具调用时才返回 `tool_calls`。`MALFORMED_FUNCTION_CALL`、
  `UNEXPECTED_TOOL_CALL`、`TOO_MANY_TOOL_CALLS`、`OTHER`、`LANGUAGE` 等异常原因返回
  `abnormal_finish` 错误，不会映射为 `stop` 或 `tool_calls`。
- `promptFeedback.blockReason` 非空时一律返回 `finish_reason: "content_filter"`，不依赖具体的
  block reason 枚举值，也不会伪装成正常空回答。
- Gemini part 的 `thoughtSignature` 会通过工具调用的 `thought_signature` 扩展字段返回；客户端在
  后续 assistant `tool_calls` 中带回同名字段（也接受 `thoughtSignature`）时，会恢复到对应
  `functionCall` part，完成需要签名的工具调用往返。
- `usageMetadata`：`promptTokenCount` -> `prompt_tokens`，
  `candidatesTokenCount + thoughtsTokenCount` -> `completion_tokens`，
  `thoughtsTokenCount` 同时写入 `completion_tokens_details.reasoning_tokens`；
  `cachedContentTokenCount` 写入 `prompt_tokens_details.cached_tokens`；
  `totalTokenCount` -> `total_tokens`。SSE 路径的 token 计数字段按 JSON 整数语法校验，
  只接受可选负号、`0` 或非零开头的十进制数字；`+1`、`01`、`-01` 和溢出的整数会被拒绝，
  与普通 JSON 路径行为一致。
- 流式：逐条转成 `chat.completion.chunk`，每个 chunk 的文本仍然原样输出；图片以 `image_url`
  content part 数组出现在 `delta.content` 中；最后发送一个带 `finish_reason` 的 chunk 和
  `data: [DONE]`。若上游返回 usage，会附在最后一个 finish chunk 上。异常 finishReason 会发送
  `abnormal_finish` 错误事件并以 `[DONE]` 结束，不会补造正常完成。
- 流式解析按 SSE 事件（空行分隔）组合同一事件内的多条 `data:` 行后再解码，支持合法的多行
  JSON，并支持 CR、LF、CRLF 三种行结束符。事件收集使用可溢出缓冲：事件超过
  `sse_spill_threshold_bytes` 后先写入临时文件，不再先构造完整的 Scanner 行和事件字符串。
  大文本和 `inlineData.data` 在 JSON 解析时增量解码到可溢出载荷存储，随后再增量写出或聚合。
  EOF 前没有收到 `[DONE]`、`finishReason` 或 `promptFeedback.blockReason` 时按不完整流报错；
  即使收到了 `[DONE]`，如果此前没有有效候选或 prompt block，也会发送
  `invalid_upstream_response` 错误事件并以 `[DONE]` 结束，不会补造正常的 `stop` 结束。

## `aggregate_sse`：非流式请求走上游 SSE 聚合

适用场景：客户端要求非流式响应，但希望本服务从上游的流式接口读取并聚合，避免某些上游/网络对
长非流式请求的限制。

三个入口，优先级从高到低：

1. 请求体字段：`"aggregate_sse": true`
2. 查询参数：`?aggregate_sse=true`
3. 请求头：`X-Aggregate-SSE: true`

优先级按参数是否出现判断，而不是按值是否为空判断：显式空的查询参数 `?aggregate_sse=` 视为
`false`，不会继续读取请求头；请求体中的空字符串同样视为 `false`。

示例：

```bash
# 请求体
curl -s http://127.0.0.1:8080/newapi1/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-2.0-flash","aggregate_sse":true,"messages":[{"role":"user","content":"hi"}]}'

# 查询参数
curl -s 'http://127.0.0.1:8080/newapi1/v1/chat/completions?aggregate_sse=true' ...

# 请求头
curl -s http://127.0.0.1:8080/newapi1/v1/chat/completions \
  -H 'X-Aggregate-SSE: true' ...
```

行为：

- 只有 `aggregate_sse` 为真且 `stream` 不为 `true` 时才触发。
- `stream=true` 时始终走正常流式，`aggregate_sse` 被忽略（不解析、也不校验其取值）。
- 触发后，本服务请求上游的 `streamGenerateContent?alt=sse`，逐条读取 SSE，把所有文本块
  原样拼接（不加换行），最后返回一次普通的 OpenAI 非流式响应。
- 零长度文本 part（例如 `{"text":""}`）不写入任何内容，聚合路径与流式路径一致；空文本独占
  事件、空文本终止事件（`parts:[{"text":""}]` 加 `finishReason`）以及空文本后接 usage 尾事件
  都按正常完成处理，不会报 `missing stored text payload`。
- SSE 事件按行增量收集；单个事件超过 `sse_spill_threshold_bytes` 后先把事件内容写入随机命名的
  临时文件，再从该存储中流式解析。单个 SSE 事件没有固定总字节上限；大文本和图片
  `inlineData.data` 在解析时增量解码到可溢出载荷存储，聚合时再增量复制到文本/图片目标缓冲。
  落盘失败会终止聚合并返回错误，不会无限回退到内存。
- 聚合工具调用会保留 `thoughtSignature`，并受独立上限约束（最多 4096 个调用、工具调用数据累计
  16 MiB；累计字节包含 name、id、args 和 thoughtSignature）；超过时返回错误。
- 聚合图片的元数据也受独立预算约束（最多 4096 张、MIME/偏移/长度等元数据累计 1 MiB），图片
  `mimeType` 最长 255 字节；超过时返回明确错误。图片载荷本身仍写入可溢出存储，不受该元数据预算
  限制。
- 临时文件用 `os.CreateTemp` 创建在本次请求所属实例的私有目录中，文件名随机；请求结束（包括
  出错、取消）立即删除。
- 聚合完成后从可溢出存储（内存或临时文件）增量读取文本并转义为 JSON 字符串直接写出；图片载荷
  按记录的偏移和长度增量读取，边读边写 `data:` URL，不会再把整个响应或整张图片重新读回内存。
- `sse_cache_max_bytes` 是每个实例已完成/遗留缓存文件的软上限：服务启动时、运行期间每分钟、以及
  请求结束时都会触发清理，按时间从旧到新删除，直到本实例目录的总量回到上限以内。刚创建不到
  一分钟的文件不会被清理。正在聚合的文件不会被清理，也**不会因为超过该上限而截断**。文本与
  图片载荷合计超过上限时，仍会完整返回，只记录一条日志，并在完成后删除该请求的临时文件。
- 启动和定时清理会处理基目录下未被锁定、且超过宽限期的旧 `run-*` 实例目录；旧版本遗留的
  `oai-to-gemini-sse-*` 顶层目录只在启动时清理。两者都只删除本服务格式的旧溢出文件；仍在运行的
  其他实例持有目录锁，不会被误删。
- 聚合文本仍按顺序原样拼接；图片会作为 `image_url` content part 保留，并与文本的相对顺序一致。
  暂不支持或未建模的响应内容类型会让聚合返回明确错误。
- 上游流在未收到 `[DONE]`、`finishReason` 或 `promptFeedback.blockReason` 前结束会返回错误，
  不会把部分结果当作完整聚合结果。即使收到 `[DONE]`，如果此前没有有效候选或 prompt block，
  同样返回 `invalid_upstream_response`，不会把空回答当作成功。异常 finishReason 同样返回错误，
  不会映射为正常完成。

## 密钥与参数透传

- 本服务不保存密钥，配置里没有密钥字段。
- 原样转发客户端请求的 `Authorization` 头（例如 `Bearer ...`）。
- 原样转发 `x-goog-api-key` 头和查询参数里的 `key`。
- 上游重定向只跟随同源（协议、主机、端口一致）目标，且不允许 HTTPS 降级；跨源重定向不会
  被跟随，因此凭证不会发送到其他主机。
- 日志只记录方法、路径、状态码和耗时；不会记录 `Authorization`、`x-goog-api-key`、`key` 或请求/响应体。
- 除明确映射的字段外，不会凭空增删参数；Gemini 原生字段按上面的列表透传。
- 无法映射的 OpenAI 字段（例如 `n`、`logit_bias`、`user`、`stream_options`、OpenAI 的布尔
  `logprobs` 等）不会硬塞进 Gemini 请求，详见“已知边界”。

## Docker

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml 中的占位上游地址
docker build -t oai-to-gemini .
docker run --rm -p 8080:8080 \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  oai-to-gemini
```

镜像说明：

- 多阶段构建，构建阶段 `golang:1.27`，运行阶段 `scratch`。
- 复制了 CA 证书，支持 HTTPS 上游。
- 默认 `WORKDIR /app`，入口 `/oai-to-gemini`，默认读取 `/app/config.yaml`。
- 暴露 `8080`，默认监听 `:8080`。
- 镜像内不含任何密钥或真实配置。
- 以非 root 用户 `65532` 运行；挂载的 `config.yaml` 需要可读（例如 `chmod 644 config.yaml`）。

## Docker Compose 与 New API 组网

`docker-compose.yml` 定义了：

- `oai-to-gemini` 服务：构建当前仓库，挂载 `./config.yaml:/app/config.yaml:ro`，加入 `newapi`
  网络，并设置网络别名 `oai-to-gemini`。
- 一个受 `with-newapi` profile 保护的 `new-api` 示例服务，默认不会启动。
- 名为 `newapi` 的网络。

只启动中转服务：

```bash
docker compose up -d oai-to-gemini
```

同一个 `newapi` 网络里的 New API 容器可以直接用内网地址访问：

```text
http://oai-to-gemini:8080/newapi1
```

New API 渠道配置建议：

- 渠道类型选 OpenAI 兼容。
- Base URL 填 `http://oai-to-gemini:8080/<供应商名>`，例如
  `http://oai-to-gemini:8080/newapi1`。
- 密钥填上游要求的 token；New API 会把它作为 `Authorization` 发过来，本服务原样转发。
- 本服务同时接受 `/v1/chat/completions` 和 `/chat/completions`；如果某个 New API 版本会自己
  追加 `/v1`，按它实际请求的路径调整 Base URL 即可。

如果你的 New API 已经在另一个 compose 项目里运行：

1. 不要启动示例 `new-api` 服务。
2. 把本 compose 里的 `newapi` 网络改成 external，并保持网络名一致：

```yaml
networks:
  newapi:
    external: true
    name: newapi
```

3. 确保两个 compose 项目使用同一个真实网络名；`oai-to-gemini` 的网络别名仍然可用。

如果想用示例 New API 一起跑：

```bash
docker compose --profile with-newapi up -d
```

示例服务没有写入任何真实密钥；实际使用时请自行通过 New API 的环境变量/初始化流程配置。

## GHCR 云端构建

`.github/workflows/docker.yml`：

- 在 push 到 `main` 和 push 任意 tag（包括 `release/v1.0.0` 这类带 `/` 的 tag）时触发。
- 使用 `docker/metadata-action` 从 `github.repository` 推导镜像名：
  `ghcr.io/<owner>/<repo>`，不硬编码用户名。
- 权限只有 `contents: read` 和 `packages: write`。
- 构建上下文为仓库根目录。
- 使用 `GITHUB_TOKEN` 登录 GHCR 并推送。

`.github/workflows/go.yml`：

- 在 push 和 pull request 时运行。
- 执行 `gofmt -l .` 检查、`go vet ./...`、`go test ./...`。

## 代码结构

```text
main.go                         启动、配置加载、HTTP server、优雅退出
internal/config/                配置解析、别名、默认值、校验
internal/router/                供应商路由解析
internal/convert/               OpenAI <-> Gemini 请求/响应转换
internal/aggregate/             SSE 聚合缓冲、文件溢出、软上限清理
internal/proxy/                 HTTP 代理、SSE 解析/转发、错误包装、models 兼容
```

## 本地验证结果

在仓库根目录依次执行：

```bash
gofmt -l .
go vet ./...
go test ./...
```

当前结果：

- `gofmt -l .`：无输出（通过）。
- `go vet ./...`：通过。
- `go test ./...`：全部通过。

测试覆盖包括：

- 多文本块拼接不加换行（OpenAI -> Gemini、Gemini -> OpenAI、流式、聚合）。
- system / user / assistant / tool 消息转换；同一轮并行工具结果合并为一个 user 回合，不同轮次
  保持独立。
- `generationConfig` 映射及不补默认值。
- 供应商路由存在/不存在；路由不会对已解码路径二次解码，配置拒绝含 `%` 的供应商名。
- `aggregate_sse` 解析函数覆盖请求体、查询参数、请求头三种入口及优先级；代理集成测试分别验证仅查询参数和仅请求头作为唯一入口时的完整聚合行为（上游 `streamGenerateContent` + `alt=sse`、`aggregate_sse` 不透传、最终返回一次非流式 OpenAI JSON 响应）。
- SSE 聚合溢出到文件且内容完整。
- 用较小的测试用软上限验证“上限不是截断上限”，活动文件不会被清理。
- 已完成缓存文件按时间淘汰；默认目录会在启动时清理旧的 `oai-to-gemini-sse-*` 遗留目录，但保留
  近期目录和含非本服务文件的目录。
- 多个实例共用基目录时使用独立实例目录和目录锁，清理不会删除其他实例仍在使用的活动文件。
- 聚合工具调用和图片元数据的数量/字节预算（工具调用累计字节包含 name、id、args 和
  `thoughtSignature`），图片 `mimeType` 长度限制，以及超限后的临时文件清理。
- 代理层：未知供应商 404、Authorization 透传、跨源重定向不转发密钥、非流式/流式/聚合端到端。
- 同源 307/308 重定向重放请求体时，发送超时按每一跳重新计时；测试等到目标接连接待重放请求后
  才开始计算发送阶段，避免把 16 MiB 请求的读取、解析、转换和编码耗时算进发送超时；目标不读
  body 时请求仍能及时超时退出并释放资源，正常同源重定向后的长 SSE 仍完整返回。非重定向的
  “上游接受连接但不读 body”测试同样等到本地监听器接受连接后才开始发送超时断言，预处理阶段
  使用独立的等待窗口，避免 `-race` 下把预处理耗时误算为发送超时失败。
- SSE 多行 `data:` 事件、CR/LF/CRLF 行结束符、提前 EOF 不完整流、工具调用 `finish_reason`、
  安全拦截、thought token usage、流式请求忽略非法 `aggregate_sse`。
- 单个大 SSE 事件在完成前就触发溢出，随后仍完整返回；累计响应超过 64 MiB 的完整性测试继续保留。
- 浅层但极宽的 `parts` 数组在解析阶段按有界数量终止，代理层用例验证随后清理已创建的临时文件；
  超多未知字段也在解析阶段按有界数量终止。请求顶层大量不同未知字段或重复字段在预算边界处被
  拒绝，测试同时断言拒绝发生在顶层 `map` 大量分配之前；边界内的合法透传字段仍完整保留。
  `generationConfig` 嵌套对象另有同量级（4096）的字段预算，大量不同键或重复键同样在嵌套
  `map` 分配前被拒绝；边界内的嵌套字段仍透传到 Gemini 生成配置。
- SSE 与普通 JSON 路径对 `+1`、`01`、`-01`、合法负数、`-0`、小数/指数和整数溢出给出相同的
  成功/失败结果。
- 超长数字、超长非法字面量和超长整数 token 在跳过、原样复制和整数解析路径上都在有界读取量内
  被拒绝，不会把完整 token 先分配进内存；代理层超长字面量用例还验证了临时文件清理。
- 模型列表缺 `id`/`name`、空标识、非字符串标识、超长标识、跨分页累计字节超限、超大分页令牌
  以及有效/无效条目混合都返回 502；合法空列表仍成功。模型列表顶层响应按只含 `models`、`data`、
  `error`、`nextPageToken` 的类型化结构解码，未知顶层字段不会展开为 `map` 条目；空 `models`
  列表附带 10 万个未知顶层字段时分配有上界。OpenAI 格式第一页仍用原始响应字节原样透传。
  未支持响应内容在非流式、流式和聚合路径都断言 `unsupported_response` 错误码。
- `aggregate_sse` 的查询参数显式空值不会回退到请求头。
- SSE JSON 嵌套超过 1000 层时返回有界错误并清理临时文件；非法 UTF-8 文本被拒绝，写出层只保留
  跨读取边界的未完成字符，非法字节不会让 pending 累积整个响应。
- SSE 无效结构（`{}`、`null`、空候选、仅 `[DONE]`）在流式和聚合路径返回
  `invalid_upstream_response`；合法空内容终止候选和 usage-only 尾事件可正常完成。聚合路径
  还覆盖空文本独占事件、空文本终止事件以及空文本后接 usage 尾事件，均返回正常完成。
- 退出生命周期：聚合已落盘时 `Shutdown` 等待活动请求完成后再删除缓存目录；超过期限时取消
  活动请求、等待 handler 释放资源后再关闭缓存存储。
- 多候选请求拒绝、任意非空 `blockReason` -> `content_filter`、异常终止优先于工具调用。
- `thoughtSignature` 工具调用往返、多模态 part 映射、未支持 part 报错、模型列表分页合并。
- 代理层超过缓存软上限仍完整返回，成功/错误/取消后临时文件清理。
- 缓冲层对 65 MiB 非均匀内容做增量哈希比对；代理层用超过默认 64 MiB 的端到端用例校验完整内容
  与临时文件清理。
- 图片独占和文本/图片混合的响应转换有单元测试覆盖；代理层覆盖非流式与聚合的混合响应、流式图片
  响应，以及聚合大图片落盘后完整返回；暂不支持或未建模的非文本类型返回错误。
- 工具结果字符串与文本 content part 数组等价（`"alphabeta"`），工具结果中的非文本块报错。
  工具结果 content 数组的 content part 识别使用有界结构扫描，只检查每个元素顶层是否有 `type`
  键，不再为类型判断构造完整 `map`；10 万字段级宽业务对象（不同键和重复键）仍走原有业务结果
  包装分支，识别阶段分配有上界；仅嵌套对象含 `type` 时不会误判，含顶层 `type` 的正常 content
  part 数组仍按原规则拼接文本。
- 异常 finishReason（含工具调用错误）在非流式、流式和聚合路径返回错误。
- 停滞上游触发空闲超时；慢速下游写触发写空闲超时；下游持续成功但较慢地写出时，上游读取空闲
  计时不会误触发，后续内容、usage、finish chunk 和 `[DONE]` 仍完整返回。

## 已知边界

- **图片/音频/文件内容**：请求侧支持 `image_url`、`input_audio`、`file` 的常见 base64 data URL 和
  HTTP(S) URL 映射；`file.file_id` 等无法映射的值会返回明确错误，不会静默删除或伪造内容。
  响应侧目前只支持 Gemini 图片 `inlineData` 转 `image_url` content part；其他非文本输出返回明确错误。
- **工具**：只支持 `type: "function"` 的 tools。OpenAI 内置工具（如 web search）会被忽略。
  `parameters` JSON Schema 原样透传，Gemini 不支持的 schema 关键字可能导致上游报错。
  Gemini 流式 `functionCall` 按完整调用输出，不拆分 arguments 片段。
- **thought parts**：Gemini 返回的 `thought: true` 文本不会混入 OpenAI `content`。
- **多候选**：响应侧只支持单候选；`candidateCount > 1` 和 OpenAI 的 `n > 1` 会在请求上游前
  返回错误，`n: 1` 按单候选处理。
- **未映射的 OpenAI 参数**：`logit_bias`、`user`、`stream_options`、`parallel_tool_calls`、
  OpenAI 布尔型 `logprobs`、`top_logprobs`、`service_tier`、`store`、`metadata` 等不会发送给 Gemini。
  如果上游支持对应 Gemini 原生字段，可以直接用 Gemini 风格字段名传入。
- **finishReason**：区分 `STOP`、`MAX_TOKENS`、安全相关原因和工具调用；长度限制和安全拦截优先于
  工具调用。异常原因（如 `MALFORMED_FUNCTION_CALL`、`UNEXPECTED_TOOL_CALL`、`OTHER`）返回
  `abnormal_finish` 错误，不映射为 `stop`。
- **模型名**：应传裸模型名；包含 `/` 的模型名会被 URL 转义，可能不符合上游路径预期。
- **上游错误**：非 2xx 状态码原样返回，错误体包装成 OpenAI 格式；不会把上游敏感头写入日志。
  SSE 文本中的非法 UTF-8 按上游协议错误处理；写出层的兜底替换只保留跨读取边界的未完成字符。
- **资源上限**：请求体原始字节上限 64 MiB（读取到上限后拒绝，不截断转发）；解析请求前会在
  结构扫描阶段把顶层对象字段数限制为 4096（包括未知字段，重复字段也计数），超限请求在解码为
  `map` 前被拒绝；嵌套的 `generationConfig` 对象也限制为 4096 个字段（同样包括未知字段，
  重复字段也计数），超限请求在创建该嵌套 `map` 前被拒绝。常规 `generationConfig` 只有少量
  字段，不受影响。工具结果 content 数组的类型识别使用有界结构扫描，不按字段数构造 `map`。
  非聚合上游 JSON 响应上限 128 MiB，models 单页响应上限 16 MiB（最多 100 页、10 万个模型、
  单个模型标识 8 KiB、跨页累计模型名字节 2 MiB、单个分页令牌 8 KiB、累计令牌字节 1 MiB），
  models 顶层对象按类型化结构解码（未知顶层字段不展开为 `map` 条目），SSE JSON 嵌套最多
  1000 层；超过时返回错误，不会截断后当作成功。单个 SSE 事件不再有固定总字节上限，可落盘的
  文本与图片载荷可以超过 64 MiB 并完整返回；解析期对不可落盘的元数据另有数量与累计字节预算
  （每事件最多 256 个候选、16384 个 part、每个 part 最多 1024 个未知字段、未知字段总数最多
  65536，非载荷元数据累计最多 64 MiB）；数字/布尔/null 字面量在读取过程中增量校验，被跳过
  的字面量最多 1 MiB 且不会在内存中完整构造，需要保留的字面量按写入字节计入上述 64 MiB
  元数据预算，整数 token 最多 32 字节。聚合文本与图片载荷的总大小仍只受
  `sse_cache_max_bytes` 软上限影响，不会因该软上限被截断；聚合工具调用和图片
  元数据另有独立的数量/字节上限，图片 `mimeType` 最长 255 字节。
- **单个大 SSE 事件的内存**：事件收集、文本和图片大字段解析都使用可溢出存储与固定大小的解析
  缓冲，不再同时保留 Scanner 行、完整事件字符串、`Unmarshal` 字节和 part 双重解码等多份完整
  副本。事件原文和解析出的文本/图片载荷会分别写入存储（超过阈值时落盘）；每个 part 的非大字段
  元数据仍保留在内存，因此解析器对候选数、part 数、未知字段数以及非载荷元数据累计字节数设置了
  独立预算，在解码阶段超限即拒绝，而不是等整个事件解析完再检查。`text` /
  `inlineData.data` 以外的字段（例如 `functionCall.args`、`error`、`safetyRatings`）仍按普通
  JSON 值解码，但受上述元数据字节预算约束。未知字段等跳过路径和原样复制路径都逐字节校验并
  增量处理字面量，不会先把任意长度的数字或非法 token 完整读入内存；任意对象/数组的跳过与
  原样复制路径都有统一的嵌套深度上限。这里不声称单事件内存占用与事件大小完全无关，但大载荷
  本身不再在内存中形成多份完整副本。
- **超时**：上游 TCP 建连使用 10 秒拨号超时，TLS 握手 10 秒，等待响应头 60 秒；向上游发送请求
  （含请求体）按每次实际发送（包括同源 307/308 重定向后的请求体重放）使用 60 秒有界超时，每一跳
  写完后即解除，不影响随后长时间的 SSE 响应；上游响应体读取使用可续期的空闲超时（默认 60 秒），
  计时只在阻塞等待上游数据时进行，读完即暂停，不会把下游处理或写出的时间算进去。下游请求读取由
  HTTP server 的 5 分钟总读取超时限制，下游响应写出使用独立的可续期空闲超时（默认 60 秒）。因此
  流式响应只要上游持续输出、下游持续成功接收，就不会因固定总时长或正常的下游背压被中断。
- **优雅退出**：默认最多等待 10 秒让活动请求自然完成；超过期限会取消活动请求并关闭剩余连接，
  然后等待 handler 释放聚合缓冲，最后才删除本实例缓存目录。超过期限的请求可能返回错误，而不是
  继续保证响应完成。
- **本地 Docker 沙箱**：本机 Docker 守护进程受沙箱限制，无法可靠拉取 Docker Hub 基础镜像或运行容器，
  因此没有把本地 `docker build` / `docker run` 成功作为验收门槛。Dockerfile、compose 和 workflow
  已按规范编写，但容器实际启动未在本机验证。
- **共享缓存目录**：默认基目录是系统临时目录下的 `oai-to-gemini-sse`，每个进程在其中使用独立的
  `run-*` 实例子目录。手动让多个实例共用同一个 `sse_cache_dir` 也是安全的：实例目录带锁，清理器
  只处理未被锁定的旧目录。跨版本混用（例如旧版本仍直接写基目录）不在该隔离保证内。
