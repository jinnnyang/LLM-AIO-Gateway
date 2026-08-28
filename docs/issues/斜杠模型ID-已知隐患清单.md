# 上游模型 ID 含斜杠：已知隐患清单

> 状态：**未修复（已知问题，非本次任务范围）**
> 记录时间：2026-08-28 · 分支 `feature/model-metadata`
> 相关已修改动：`app/database.py` 中 5 处 provider-scoped 查询已改用 `_strip_own_prefix()`，并有 `tests/test_slash_model_lookup.py`（17 例）钉住

## 1. 背景：两种「模型 ID」

| 形式 | 例子 | 存在于 |
|------|------|--------|
| **裸上游 ID** | `des/deepseek` | `provider_models.model_id`，发往上游的 `model` 字段 |
| **composite 寻址** | `vcp/des/deepseek` | 用户请求、路由规则、白名单、日志展示 |

`vcp/` 前缀是**运行时由 `provider_id` 拼出来的寻址语法，从不落库**。

大多数上游的模型 ID 不含斜杠（`gpt-4o`），所以 `provider_id` 和 `model_id` 的边界一直是隐含的。
一旦上游 ID 自己带斜杠（`des/deepseek`、OpenRouter 风格的 `anthropic/claude-3`），这个隐含边界就失效了。

## 2. 根因：`parse_model_id` 无法区分两种形式

```python
parse_model_id("vcp/des/deepseek")  # provider='vcp'  model='des/deepseek'   ✅
parse_model_id("des/deepseek")      # provider='des'  model='deepseek'       ❌ 削掉一段
```

`ModelId.parse` 用 `raw.split("/", 1)` 只切第一刀，**这是正确的设计**：它的契约是「解析用户输入的 composite 地址」。
问题在于**有些调用点手上其实已经有 `provider_id`，却仍然让字符串自己去猜**。

对比：

| 函数 | 上下文 | 语义 |
|------|--------|------|
| `parse_model_id(raw)` | 无 provider 上下文 | 猜：首段**可能**是 provider |
| `_strip_own_prefix(provider_id, raw)`<br>（`app/database.py:1591`） | 已知 provider | 只剥自家前缀一次，其余斜杠全归上游 |

判断标准就一条：**调用点手上有没有 `provider_id`。有就不该用 `parse_model_id`。**

## 3. 实测：当前运行时行为

用 provider `vcp` + 裸 ID `des/deepseek` 建库后实测：

```
find_provider_by_model('vcp/des/deepseek') -> vcp
find_provider_by_model('des/deepseek')     -> None      ← 裸斜杠 ID 查不到
find_provider_by_model('deepseek')         -> None

适配器上游名 parse_model_id(target_model).model_name:
  target_model='vcp/des/deepseek' -> 'des/deepseek'     ✅ composite 形式下正确
  target_model='des/deepseek'     -> 'deepseek'         ❌ 裸形式下削掉

_model_allowed_by_list(['vcp/des/deepseek'], 'vcp/des/deepseek') -> True
  裸↔composite 交叉、['deepseek']、['vcp/other'] 四组 -> 全 False

preprocessor 查询: 'vcp/des/deepseek' -> 命中 ; 'des/deepseek' -> None

_target_label('vcp', 'des/deepseek')      -> 'vcp/des/deepseek'
_target_label('vcp', 'vcp/des/deepseek')  -> 'vcp/des/deepseek'   （幂等）

wildcard_match('des/*', 'des/deepseek')        -> True
wildcard_match('*', 'vcp/des/deepseek')        -> True
wildcard_match('*deepseek', 'des/deepseek')    -> True
```

**核心结论：只要 `target_model` 始终保持 composite 形式，适配器层实际是对的。**
风险窗口只有一个 —— `target_model` 被写成**裸斜杠形式**的时候。而 routing rule 的 `target_model` 是管理员手填的，填成 `des/deepseek` 完全可能。

这也是为什么本次没有顺手改这些点：它们**当前不必然出错**，改了反而需要一整套端到端场景才能验证是否改对。

## 4. 隐患清单

### A 类 · 拿到 `target_model` 但函数签名里没有 `provider_id`

风险：`target_model` 若为裸斜杠形式，发往上游的 `model` 被削成 `deepseek`，上游返回 404 / model not found。

| 位置 | 代码 | 备注 |
|------|------|------|
| `app/adapters/anthropic.py:37` | `mid = parse_model_id(model)` → `req_body["model"] = mid.model_name` | 调用方 `app/adapters/anthropic.py:261`、`app/adapters/anthropic_streaming.py:45`。**函数已有 `provider_info` 参数，可取 `provider_info["id"]`** |
| `app/adapters/responses.py:79` | `body["model"] = parse_model_id(internal.target_model).model_name` | 调用方 `:88`、`:99`。手上无 provider，需从 `internal` 补 |
| `app/services/lite_llm.py:117` | `model = parse_model_id(model).model_name` | **签名 `get_litellm_model_name(model, provider)` 已有 `provider` dict，取 `provider["id"]` 即可 —— 这是最容易修的一个** |
| `app/router/proxy.py:1547` | `model_name = parse_model_id(target.model).model_name`，随后 `model.get("id") == model_name` 比对 | `_provider_model_for_target`；**`target.provider_id` 可用**（`RouteTarget` 有独立字段，见 `app/core/policy.py:193-195`）。比对失败 → 查不到模型配置 |
| `app/router/proxy.py:1536` | `_target_model_for_log` | 仅日志展示，影响可读性不影响功能 |

即使是「最容易修」的 `lite_llm.py:117`，也要注意它对 `api_base` 做 `f"openai/{model}"` 拼接 —— 上游 ID 自带斜杠时拼出 `openai/des/deepseek`，liteLLM 侧如何解析需要单独验证。**不是替个函数就完事。**

### B 类 · 处理用户输入的 composite 地址 —— 语义正确，不该改

这些位置拿到的就是用户/管理员写的 composite 地址，`parse_model_id` 的契约在这里成立。

- `app/core/policy.py:270`（`_target_label`）、`:291`（fallback 匹配）、`:360`（routing rule match_scope）、`:444`（preprocess）
- `app/router/proxy.py:2578`、`:2580`、`:2593`、`:2637`、`:2645`（allow-list 匹配）、`:4138`（images）

**但**：允许裸斜杠 ID 之后，allow-list 的语义会变模糊 —— 白名单里写 `des/deepseek`，用户到底想允许「provider `des` 的 `deepseek`」还是「任意 provider 的 `des/deepseek`」？实测目前两种交叉写法都返回 `False`，即**裸斜杠 ID 无法通过白名单**。这是需要产品决策的歧义，不是能悄悄改掉的 bug。

### C 类 · `is_composite` 分支逻辑 —— 与本次的 provider-scoped 查询不同构

结构上是「有前缀走 A 路，无前缀走 B 路」，不能套用 `_strip_own_prefix`。

- `app/router/admin.py:893`（preprocessor toggle）
- `app/router/proxy.py:2373`（preprocessor 查询）
- `app/database.py:1325`（`set_model_image_generation`）

### D 类 · 图像生成链路

`image_generators.provider_model` 存的是 composite 形式，整条链路都在 parse / 重组，改动面最大。

| 位置 | 代码 |
|------|------|
| `app/router/admin.py:1030` | `provider = get_provider(mid.provider_id) if mid.provider_id else find_provider_by_model(mid.model_name)`；`valid_ids` 比对 `{mid.model_name, f"{mid.provider_id}/{mid.model_name}"}` |
| `app/adapters/imagegen.py:234` | `request_model = model or config.get("model") or parse_model_id(config.get("provider_model") or "").model_name` |
| `app/router/proxy.py:229` | `_resolved_image_generator`：`resolve_provider(image_mid.model_name, image_mid.provider_id)` → `resolved["model"] = image_mid.model_name` |

### E 类 · 其他

- `app/database.py:1907` `find_provider_by_model` —— 裸斜杠 ID 查不到（实测 `None`）。`mid.provider_id` 分支用 `m.model_id = mid.model_name` 比对，无 provider 前缀时无从判断
- `app/router/admin.py:350`、`:365`、`:671`

## 5. 已修部分为何是安全的

`app/database.py` 那 5 处（`get_model_image_generation` / `get_model_responses_capability` / `set_model_responses_capability` / `update_model_responses_capability` / `update_model_responses_tool_types`）能单独修完并验证，因为：

1. 它们都是 **provider-scoped** 的 —— 函数签名里就有 `provider_id`，无需改调用契约
2. 影响面止于单表读写，不跨越请求链路
3. 可以用单元测试完整覆盖

反向验证做过：把修复回滚后跑 `tests/test_slash_model_lookup.py`，**恰好 4 例失败**，全是「断言截断行不该被碰」的用例（报 `assert '["web_search"]' == '[]'`），另 13 例前后都过。

这说明旧代码是被 SQL 里 `model_id IN (?, ?) OR model_name = ?` 的 **OR 兜住**的（第一个 `?` 传的是未加工的 `model`）——
**读取路径侥幸正确，写入路径会连带写坏同 provider 下真实存在的 `deepseek` 行。**

## 6. 如果要跟进

这是「斜杠 ID 端到端跑通请求」的独立课题，建议单独立项，而不是补丁式逐点替换。

建议顺序：

1. **先定语义**：`target_model` 在内部流转时统一用 composite 还是裸形式？现在两种都可能出现，这是所有 A 类问题的根 —— 定死一种，多数问题自动消失
2. **B 类的 allow-list 歧义要产品决策**，不是技术选择
3. 补端到端测试：建一个裸 ID 含斜杠的 provider，跑通 chat / stream / anthropic / responses / images / preprocessor / fallback / allow-list 全链路
4. 再按 A → D → C 顺序改，每步都有失败测试兜底

### 不建议做的事

- **全局把 `parse_model_id` 换成 `_strip_own_prefix`** —— B 类会直接坏掉，那里没有 `provider_id` 可传，语义本来就是「猜」
- **在 `ModelId.parse` 里加"智能"判断**（比如查 DB 判断首段是不是真 provider）—— 会让纯函数依赖数据库，且引入首段恰好与某 provider 同名的竞态

## 附：相关设计约束

- `provider_models` 有 `UNIQUE(provider_id, model_id)`，同 provider 下裸 ID 唯一
- rename 端点已能正确处理斜杠 ID：`_strip_own_prefix` + `foreign_provider_prefix` 错误码（新 ID 首段命中**真实存在的** provider ID 时拒绝，避免"改名"变成"搬 provider"）
- `%2F` 会被 ASGI 服务器在**路由匹配之前**解码，`encodeURIComponent` 救不了 → 故 rename 另有 body 承载旧 ID 的端点 `PUT /admin/providers/{provider_id}/models/rename`，其余端点改用 `{model_id:path}`
