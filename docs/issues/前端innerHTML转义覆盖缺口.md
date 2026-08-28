# 前端 `innerHTML` 转义覆盖缺口

> 状态：**未收敛（守卫范围有限，非已知失败）**
> 记录时间：2026-08-28 · 发现于 PR #5 收尾复核
> 相关测试：`tests/test_metadata_api.py::test_static_js_escapes_candidate_values`（当前**通过**）

---

## 1. 先纠正一处错误记录

PR #5 的「已知遗留」第一条写：

> `tests/test_metadata_api.py::test_static_js_escapes_candidate_values` 失败 —— 报出约 30 处既有代码…

**这条是错的。** 实测该测试通过，全量 882 passed / 0 failed。

错误来源：该测试**最初**确实是全文扫描 `app.js` 的所有 `.innerHTML` 赋值，会报出约 30 处既有渲染代码。但在同一个 commit（`6d5763c`）里它已被收窄，收窄后不再失败。撰写 PR 描述时沿用了收窄前的观察，未复测。

真实遗留问题不是「测试失败」，而是下面的**覆盖缺口**。两者性质不同：前者是噪音，后者需要立项。

## 2. 当前守卫的实际范围

`tests/test_metadata_api.py:552`：

```python
candidate_terms = ("candidate", "metadata", "capabilit", "sync")
if ".innerHTML" not in stripped or "=" not in stripped:
    continue
if not any(term in stripped.lower() for term in candidate_terms):
    continue
```

即：**只检查行内文本含 candidate / metadata / capabilit / sync 的 `innerHTML` 赋值**。判定为安全的情形（`:557-568`）：右值是字符串字面量、是 `''`/`null`/`0` 等常量、或含 `escapeHtml` / `escHtml` / `jsEsc` / `sanitize` / `textContent` / `DOMPurify`。

这个收窄是**有意为之**，理由写在 `:543-551` 的注释里：既有 admin builder（`providerFormHtml`、`.map(...)` 链、本地拼接的 `html` 字符串）内部自行转义，全文扫描无法把它们和真正不安全的代码区分开。

收窄本身合理 —— 它让守卫精确对应设计方案 §6 用例 8（外部源候选值回填）。问题在于**收窄之后没有任何东西覆盖剩下的部分**。

## 3. 缺口量化

`app/web/static/app.js`（约 4500 行，无构建、无框架、无模块系统）：

| 指标 | 数量 |
|---|---|
| `.innerHTML` 出现次数 | 61 |
| `escHtml(` 调用次数 | 143 |
| 被上述测试覆盖的 `innerHTML` 赋值 | 仅含 4 个关键词的那部分 |

`escHtml` 用得不少，说明项目**有**转义意识；但调用密度和 `innerHTML` 点位不是一一对应关系，无法据此断言全覆盖。未被守卫的渲染路径包括 provider 列表、routing 规则表、fallback 链、stats 图表标签、logs 表格等。

这些路径的数据源多为本地配置与网关自身记录，可信度高于 models.dev / OpenRouter 这类外部目录，风险等级低于元数据候选值 —— 这也是当初优先给元数据加守卫的原因。但「风险较低」不等于「已验证安全」，尤其 `request_logs` 里会落入上游返回的错误文本。

## 4. 若要收敛，建议路径

不建议直接把 `candidate_terms` 过滤删掉再逐行修 —— 那会让一个本来精确的测试变成 30 条待办的看板，且每次前端改动都可能扰动断言。

更可行的顺序：

1. **先分类，不先修**。把 61 处 `innerHTML` 按数据来源分三档：纯本地字面量 / 网关自身数据 / 含上游或用户输入。只有第三档需要动。
2. **给第三档补测试，一处一例**，用真实恶意载荷断言输出被转义，而不是靠扫源码文本推断。源码扫描只能证明「调了转义函数」，不能证明「转义正确」。
3. **考虑统一渲染入口**。当前是字符串拼接 + `innerHTML`，全项目同一范式。若引入一个 `renderRow(tpl, data)` 之类的收口函数并强制走它，守卫可以退化为「禁止裸 `innerHTML` 赋值」这一条 lint 规则，比逐点扫描稳固得多。这属于前端重构，规模远超本议题，需独立评估。

## 5. 不建议做的事

- **不要**为了让全文扫描通过而给所有 `innerHTML` 无脑套 `escHtml` —— 有些位置传入的是刻意构造的 HTML 片段（如带 `<strong>` 的组合内容），套上转义会把标签显示成字面量，直接破坏界面。
- **不要**把这条测试改成 `xfail` 或加 `# noqa` 式豁免 —— 它现在是通过的，改成豁免等于把一个有效守卫降级成装饰。
- **不要**依赖「`escHtml` 调用了 143 次」这类计数作为安全结论。分母不明的比率没有意义。

## 6. 关联

- 设计方案 §6 用例 8 与 §5.1「候选回填」：`docs/plans/模型能力元数据扩展-设计方案.md`
- 专家评审第 8 条（XSS 列为本设计最大安全风险）：`docs/plans/模型能力元数据扩展-专家评审.md:51`
