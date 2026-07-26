# HIVE-K — Agentic System (Python)

Hệ thống agentic có trạng thái cho HIVE-K: nhận yêu cầu bằng tiếng Việt, dựng ngữ cảnh
từ dữ kiện có nguồn, lập kế hoạch nội dung, viết bài theo từng nền tảng, kiểm tra rủi
ro, dừng lại để người dùng duyệt, rồi **học từ chỉnh sửa của người dùng**.

Đây không phải một lớp bọc quanh lời gọi LLM. Phần lớn hệ thống là mã xác định; mô hình
ngôn ngữ chỉ được gọi ở những nút thật sự cần hiểu/ sinh ngôn ngữ.

---

## 1. Chạy trong 30 giây

Không cần MongoDB, không cần API key. Hệ thống tự hạ cấp xuống store in-memory + mock LLM.

```bash
cd apps/agentic-system
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/Scripts/python.exe -m uvicorn hivek_agent.api.app:app --port 8100   # Windows
# source .venv/bin/activate && uvicorn hivek_agent.api.app:app --port 8100  # macOS/Linux
```

Kiểm tra:

```bash
curl http://localhost:8100/health
# {"status":"ok","storeBackend":"memory","llmProvider":"mock","skillsLoaded":3,
#  "warnings":["Using in-memory store - data will not persist across restarts.", ...]}
```

`/health` luôn nói thật đang chạy bằng gì. Nếu thấy `mock` hoặc `memory` mà bạn không
muốn vậy, xem mục 3.

Tài liệu API tự sinh: <http://localhost:8100/docs>

## 2. Chạy full flow demo

```bash
.venv/Scripts/python.exe -m pytest -q          # test không cần mạng
```

Luồng hoàn chỉnh (mỗi bước là một endpoint thật):

```
1. POST /v1/chat/messages  "viết bài cho facebook"
   -> status=needs_user_input. Hệ thống TỪ CHỐI viết vì thiếu dữ kiện bắt buộc,
      và nói rõ đã tìm ở đâu + trả về widget để bổ sung.

2. POST /v1/setup/social | /v1/setup/brand | /v1/setup/drive
   -> readiness_score tăng dần 0.22 -> 0.55 -> 0.75 (tính bằng rule, không do LLM bịa)

3. POST /v1/chat/messages  "lên kế hoạch đăng bài 3 ngày"
   -> plan 3 node, chấm điểm MFS bằng Python, cân bằng tầng phễu

4. POST /v1/chat/messages  "viết bài cho facebook"
   -> draft + validation, status=needs_approval, asset.status=needs_review
      (KHÔNG có đường nào tự đăng)

5. POST /v1/content-assets/{id}/decision  {"decision":"edit","editedText":"..."}
   -> learnedThisTurn: [{rule:"length", status:"candidate"}]   <- MỘT lần sửa CHƯA thành luật
   -> activeRules: []

6. Sửa lần 2 cùng kiểu
   -> status="repeated", activeRules có rule    <- lúc này mới áp dụng
   -> lần 4: "stable"
```

## 3. Nâng cấp lên hạ tầng thật

Copy `.env.example` sang `.env` rồi điền:

| Muốn gì | Đặt biến |
|---|---|
| Lưu thật vào MongoDB Atlas | `MONGODB_URI=mongodb+srv://user:<mật khẩu thật>@.../` |
| Dùng Gemini thật | `GEMINI_API_KEY=...` |
| Ép dùng mock để demo/offline | `AI_AGENT_PROVIDER=mock` |

> **Bẫy thường gặp:** Atlas đưa cho bạn chuỗi kết nối chứa literal `<db_password>`.
> Phải thay bằng mật khẩu thật. Hệ thống phát hiện placeholder này và từ chối dùng
> (nếu không, lỗi auth chỉ nổ ra ở truy vấn đầu tiên chứ không phải lúc khởi động).

### Về collection trong MongoDB

Service này **dùng chung database `hivek` với backend chính** nhưng chỉ đụng vào các
collection có tiền tố `agentic_`. Nó không bao giờ đọc/ghi `users`, `campaigns`,
`influencers`… của backend.

```
agentic_runs, agentic_node_runs, agentic_events, agentic_threads, agentic_audit
agentic_assertions, agentic_entities, agentic_edges, agentic_conflicts
agentic_brand_profiles, agentic_voice_profiles
agentic_plans, agentic_assets
agentic_feedback, agentic_preferences, agentic_edit_events, agentic_performance
agentic_checkpoints, agentic_checkpoint_writes   (LangGraph)
```

### Về hạn mức Gemini

Key free tier thường có **quota = 0 cho các model Pro** trong khi Flash vẫn chạy. Model
router có sẵn `fallback_chain`, nên tác vụ tầng Pro sẽ tự hạ xuống Flash thay vì lỗi.
Nếu key của bạn không có Pro, đặt `GEMINI_MODEL_STRATEGY=gemini-2.5-flash` cho gọn.

## 4. Nối với web UI

Client Next.js gọi thẳng service này từ trình duyệt, nên cần CORS (đã bật sẵn cho
`localhost:3000`).

```bash
# apps/client/.env
NEXT_PUBLIC_AGENTIC_PY_URL=http://localhost:8100
```

Response dùng **camelCase** để khớp type TypeScript có sẵn; dữ liệu lưu trong Mongo vẫn
là snake_case (xem `domain/base.py` giải thích cách alias hoạt động).

Nếu service Python không chạy, `use-ai-chat` tự quay về service mock cũ — demo không vỡ.

## 5. Kiến trúc

```
                    ┌──────────────── FastAPI ────────────────┐
   Next.js ──HTTP──▶│ /v1/chat  /v1/setup  /decision  SSE     │
   (browser)        └────────────────┬────────────────────────┘
                                     ▼
                         ┌──── AgenticService ────┐   run lifecycle, idempotency,
                         │  (service.py)          │   events, feedback loop
                         └───────────┬────────────┘
                                     ▼
     START ▶ authenticate ▶ route ─┬─▶ handle_setup ─────────────────────▶ END
                                   ├─▶ load_knowledge ▶ validate_required_facts
                                   │        │                    │
                                   │        │              (thiếu blocking)
                                   │        │                    └────────▶ END
                                   │        ▼
                                   │   compile_context ─┬─▶ create_content_plan ▶ END
                                   │                    └─▶ generate_draft
                                   │                              ▼
                                   │                        validate_draft ▶ END
                                   ├─▶ analyze_performance ──────────────▶ END
                                   └─▶ handle_smalltalk ─────────────────▶ END
```

LangGraph lo phân nhánh + checkpoint. Các node là hàm Python thuần trong `nodes.py`
(không import LangGraph), nên test được trực tiếp và không bị khoá vào framework.

```
src/hivek_agent/
├── config.py            # settings; mọi thứ có default an toàn
├── domain/              # Pydantic contracts (base.py giải thích alias camelCase)
├── repositories.py      # workspace_id là tham số bắt buộc -> cách ly tenant bằng cấu trúc
├── infrastructure/
│   ├── store/           # base (Protocol) | memory | mongo ($graphLookup thay Neo4j)
│   └── llm/             # base (Protocol) | mock (deterministic) | gemini | CachingLLM
├── knowledge/           # facts.py (provenance, precedence, conflict) | brand_profile.py (gap detector)
├── content/             # planner (MFS, rule-based) | composer (LLM) | validator (deterministic-first)
├── learning/            # edit_analysis.py: diff -> candidate -> repeated -> stable
├── agentic/
│   ├── nodes.py         # các node thuần
│   ├── graph.py         # dây nối LangGraph
│   ├── context_compiler.py  # cổng bắt buộc trước mọi lời gọi model
│   ├── model_router.py  # task -> tier rẻ nhất làm được + fallback chain
│   ├── skills.py        # đọc apps/server-ai/SKILL/*.md
│   └── tools.py         # registry + quyền
└── service.py, api/
```

## 6. Những ràng buộc được ép bằng mã (không phải bằng lời hứa)

| Nguyên tắc | Ép ở đâu |
|---|---|
| Không tự đăng bài | `validate_draft` luôn đặt `needs_review`; không endpoint nào publish. `publishing.*` có `requires_human_approval=True` và không intent nào thấy nó |
| Không bịa dữ kiện | Composer chỉ đọc `CompiledContext`; validator chặn `unknown_fact_reference` + `unsupported_number` |
| Không ghi đè dữ kiện đã xác nhận | `facts.py` tạo `conflict`, giữ cả hai, không xoá lịch sử |
| Một lần sửa không thành luật | `promote_preferences`: cần lặp lại ≥2 lần (hoặc user pin) mới `repeated` |
| Không dùng LLM cho việc code làm được | Router keyword trước; MFS/readiness tính bằng Python |
| Không nhét cả tài liệu vào prompt | `ContextCompiler` + `guidance(max_chars=...)`; phần bỏ đi ghi vào `omitted_sections` |
| Không rò dữ liệu giữa workspace | Repository bắt buộc `workspace_id` |
| Không log secret | `NodeRun` chỉ giữ ID, token count, latency |

## 7. Chưa làm (TODO có chủ đích)

Đánh dấu rõ để không ai tưởng đã có:

- **Temporal** — blueprint xếp vào Sprint 7 (publish/OAuth/KPI). Chưa có tác dụng phụ
  ngoài hệ thống nào nên chưa cần. Ranh giới đã sẵn: mọi tác dụng phụ nằm sau tool
  registry với `requires_human_approval`.
- **Connector thật (Drive/website/Meta)** — `SourceRef.source_type` đã có sẵn các giá
  trị; hiện mới có nguồn `user_input` qua chat.
- **Vector search** — `BRAND_MEMORY_VECTOR_INDEX` đã đặt chỗ; retrieval hiện là
  structured facts + ví dụ đã duyệt (đủ cho MVP, chưa cần embedding).
- **Ranker học máy / trend detection** — planner đang dùng weighted score; interface
  `score_breakdown` đã sẵn để thay bằng LightGBM khi có dữ liệu.
- **Auth thật** — xem `TODO(auth)` trong `nodes.py`; scope hiện cấp theo workspace.

## 8. Thêm mới thế nào

- **Thêm góc nội dung**: thêm một dòng vào `ANGLE_LIBRARY` (`content/planner.py`).
  Không đổi prompt, không đổi model.
- **Thêm skill**: bỏ một thư mục `SKILL.md` mới vào `apps/server-ai/SKILL/`, khai báo
  trong `_TASK_SKILLS` (`agentic/skills.py`).
- **Thêm luật kiểm tra**: thêm vào `RISKY_CLAIM_PATTERNS` hoặc một hàm `_check_*` thuần
  trong `content/validator.py` (nhớ: pattern viết có dấu, được fold lúc import).
- **Đổi store**: implement `DocumentStore` Protocol.
- **Đổi model provider**: implement `LLMGateway` Protocol.

## 9. Deploy

```bash
# Docker — build từ REPO ROOT vì cần copy apps/server-ai/SKILL
docker build -f apps/agentic-system/Dockerfile -t hivek-agentic .
docker run -p 8100:8100 --env-file apps/agentic-system/.env hivek-agentic

# Heroku/Railway/Render: đã có Procfile, service tự đọc $PORT
```

Yêu cầu tối thiểu: Python 3.12. Không cần Docker để chạy local.
