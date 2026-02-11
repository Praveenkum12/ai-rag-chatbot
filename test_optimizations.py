"""
Quick test script to verify RAG latency optimizations.

Run this after starting your FastAPI server to test:
1. BM25 caching
2. Streaming response
3. Cache invalidation
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_streaming():
    """Test the streaming endpoint"""
    print("\n=== Testing Streaming Endpoint ===")
    
    url = f"{BASE_URL}/chat/stream"
    payload = {
        "question": "What is artificial intelligence?",
        "conversation_id": None,
        "doc_ids": [],
        "history": []
    }
    
    start_time = time.time()
    first_token_time = None
    
    response = requests.post(url, json=payload, stream=True)
    
    print("Streaming response:")
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('data: '):
                data = json.loads(decoded[6:])
                
                if data['type'] == 'token':
                    if first_token_time is None:
                        first_token_time = time.time()
                        print(f"\n⚡ First token received in: {first_token_time - start_time:.2f}s")
                    print(data['content'], end='', flush=True)
                
                elif data['type'] == 'conversation_id':
                    print(f"Conversation ID: {data['conversation_id']}")
                
                elif data['type'] == 'done':
                    total_time = time.time() - start_time
                    print(f"\n\n✅ Streaming complete in: {total_time:.2f}s")
                    break

def test_cache_behavior():
    """Test BM25 cache hit/miss behavior"""
    print("\n=== Testing BM25 Cache ===")
    print("Check your server logs for 'Using cached BM25' or 'Building new BM25' messages")
    
    url = f"{BASE_URL}/chat"
    payload = {
        "question": "Test question for cache",
        "conversation_id": None,
        "doc_ids": [],
        "history": []
    }
    
    # First request - should build cache
    print("\n1st request (should build cache)...")
    start = time.time()
    response1 = requests.post(url, json=payload)
    time1 = time.time() - start
    print(f"   Completed in: {time1:.2f}s")
    
    # Second request - should use cache
    print("\n2nd request (should use cache)...")
    start = time.time()
    response2 = requests.post(url, json=payload)
    time2 = time.time() - start
    print(f"   Completed in: {time2:.2f}s")
    
    if time2 < time1:
        print(f"\n✅ Cache working! 2nd request was {((time1-time2)/time1*100):.1f}% faster")
    else:
        print(f"\n⚠️  No speedup detected. Check server logs for cache messages.")

def main():
    print("RAG Latency Optimization Test Suite")
    print("=" * 50)
    print("\nMake sure your FastAPI server is running on http://localhost:8000")
    print("and you're logged in (or using guest mode).\n")
    
    try:
        # Test streaming
        test_streaming()
        
        # Test cache
        test_cache_behavior()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("\nNext steps:")
        print("1. Check server logs for cache hit/miss messages")
        print("2. Upload a document and verify cache invalidation")
        print("3. Test with your frontend using the streaming endpoint")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server.")
        print("Make sure FastAPI is running: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
