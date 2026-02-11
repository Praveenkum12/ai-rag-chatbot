# Example Console Output with Timing Logs

When you make a chat request, you'll now see detailed timing information in the console:

```
============================================================
⏱️  REQUEST START: What is artificial intelligence?...
============================================================
DEBUG: Final user_id for this request: tony
⏱️  [1] Conversation setup: 45.23ms
⏱️  [2] Save user message: 12.34ms
DEBUG: Chatting as tony
DEBUG: No specific documents selected - searching ALL user documents
⏱️  [3] Build filters: 0.56ms
DEBUG: Active filters: {'user_id': {'$eq': 'tony'}}
DEBUG: Retrieved 3 docs with scores in single pass
⏱️  [4] Document retrieval: 234.56ms (3 docs)
⏱️  [5] Build context: 2.34ms (1523 chars)
DEBUG: Context length: 1523
⏱️  [6] Prepare prompt: 15.67ms
⏱️  [7] LLM call (initial): 1234.56ms
⏱️  [8] Save AI response: 18.90ms

============================================================
✅ TOTAL REQUEST TIME: 1563.16ms (1.56s)
============================================================
```

## Understanding the Timing Breakdown

| Phase                      | What It Measures                          | Expected Time                                       |
| -------------------------- | ----------------------------------------- | --------------------------------------------------- |
| **[1] Conversation setup** | Creating/loading conversation in DB       | 10-50ms                                             |
| **[2] Save user message**  | Saving your message to DB                 | 10-30ms                                             |
| **[3] Build filters**      | Constructing search filters               | <5ms                                                |
| **[4] Document retrieval** | Vector search + BM25 hybrid retrieval     | **50-300ms** (cached) / **500-2000ms** (first time) |
| **[5] Build context**      | Formatting retrieved docs for LLM         | 1-10ms                                              |
| **[6] Prepare prompt**     | Building the prompt with history/memories | 10-50ms                                             |
| **[7] LLM call (initial)** | **OpenAI API call**                       | **800-2000ms** (network + generation)               |
| **[8] Save AI response**   | Saving AI response to DB                  | 10-30ms                                             |

## What to Look For

### ✅ Good Performance

- **Total time < 2 seconds** for simple queries
- **Document retrieval < 300ms** (after first request with cache)
- **LLM call** is the biggest time sink (expected, this is network + AI generation)

### ⚠️ Performance Issues

- **Document retrieval > 500ms** consistently → Check if BM25 cache is working
  - Look for "Using cached BM25" in logs
  - If you see "Building new BM25" every time, cache might not be working
- **Total time > 5 seconds** → Likely network latency to OpenAI or large context
- **Conversation setup > 100ms** → Database might be slow

### 🔍 Cache Verification

After uploading documents, you should see:

```
First request:
DEBUG: Building new BM25 for user tony, filter a1b2c3d4
⏱️  [4] Document retrieval: 856.23ms (3 docs)

Second request (same filters):
DEBUG: Using cached BM25 for user tony, filter a1b2c3d4
⏱️  [4] Document retrieval: 123.45ms (3 docs)  ← Much faster!
```

## Bottleneck Identification

The timing logs will help you identify exactly where time is being spent:

1. **If [4] Document retrieval is slow:**
   - First time: Normal (building BM25 cache)
   - Every time: Cache not working, check invalidation logic

2. **If [7] LLM call is slow (>3s):**
   - Large context being sent
   - Network latency to OpenAI
   - Consider using streaming endpoint for better UX

3. **If [1] or [2] DB operations are slow:**
   - Database connection issues
   - Consider connection pooling

## Using the Streaming Endpoint

For even better perceived performance, use `/chat/stream`:

- First token appears in <1 second
- User sees response building in real-time
- Same backend optimizations apply
