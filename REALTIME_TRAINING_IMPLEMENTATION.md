# Real-Time ML Training with Server-Sent Events (SSE)

## Overview
Implemented real-time streaming of machine learning model results from backend to frontend as each model completes training. Instead of waiting for all models to finish, users now see results populate the leaderboard live as they're trained.

## Backend Changes (Python/FastAPI)

### 1. New Imports and Global State
```python
from fastapi.responses import StreamingResponse
import asyncio
from collections import deque

# Storage for SSE connections and model results
active_sse_connections: Dict[str, deque] = {}
model_results_queue: Dict[str, List[Dict]] = {}
```

### 2. Helper Function to Push Results
```python
def push_model_result(task_id: str, model_result: Dict):
    """Push a completed model result to the queue for SSE streaming"""
    if task_id not in model_results_queue:
        model_results_queue[task_id] = []
    
    result_event = {
        "type": "model_completed",
        "task_id": task_id,
        "model": clean_for_json(model_result),
        "timestamp": datetime.now().isoformat()
    }
    
    model_results_queue[task_id].append(result_event)
```

### 3. New SSE Endpoint
```python
@router.get("/training-stream/{task_id}")
async def stream_training_results(task_id: str):
    """Stream model results as they complete using Server-Sent Events"""
    
    async def event_generator():
        # Send connection event
        yield f"data: {json.dumps({'type': 'connected', 'task_id': task_id})}\n\n"
        
        # Stream results as they complete
        while True:
            # Check for new results
            if task_id in model_results_queue:
                for result in new_results:
                    yield f"data: {json.dumps(clean_for_json(result))}\n\n"
            
            # Check completion status
            if status == "completed":
                yield f"data: {json.dumps(completion_data)}\n\n"
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### 4. Modified Training Function
Instead of batch training with `compare_models()`, now trains models individually:

```python
# Train each model and push results immediately
for model_name in model_names:
    model = regression_module.create_model(model_name, fold=3)
    result = get_metrics(model)
    
    # 🚀 PUSH RESULT IMMEDIATELY
    push_model_result(config_id, result)
```

## Frontend Changes (React/Next.js)

### 1. SSE Connection Function
```typescript
const streamTrainingResults = (taskId: string) => {
  const eventSource = new EventSource(
    `${BACKEND_URL}/ml/training-stream/${taskId}`
  )

  const liveLeaderboard: any[] = []

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data)
    
    if (data.type === 'model_completed') {
      // Add model to live leaderboard
      liveLeaderboard.push(data.model)
      
      // Update UI in real-time
      setTrainingResults({
        leaderboard: sortedLeaderboard,
        total_models_tested: liveLeaderboard.length
      })
      
      // Show toast notification
      toast({
        title: `Model Completed: ${data.model.Model}`,
        description: `Score: ${data.model.R2.toFixed(4)}`
      })
    }
    
    else if (data.type === 'completed') {
      // Training finished
      setTrainingStatus('completed')
      eventSource.close()
    }
  }
}
```

### 2. Live Leaderboard Display
```tsx
{/* Live Results Preview */}
{trainingResults?.leaderboard?.length > 0 && (
  <Card className="border-blue-200">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Sparkles className="h-4 w-4" />
        Live Results (Top 5)
      </CardTitle>
    </CardHeader>
    <CardContent>
      {trainingResults.leaderboard.slice(0, 5).map((model, index) => (
        <div key={index} className="animate-in slide-in-from-top-2">
          <Badge>{index + 1}</Badge>
          <span>{model.Model}</span>
          <span>{model.R2.toFixed(4)}</span>
        </div>
      ))}
    </CardContent>
  </Card>
)}
```

### 3. Progress Indicator
```tsx
<Progress 
  value={trainingResults?.total_models_tested * 5} 
  className="h-2" 
/>
<p>Models Completed: {trainingResults?.total_models_tested || 0}</p>
```

## Key Benefits

### 1. **Real-Time Feedback**
- Users see results as soon as each model finishes
- No waiting for entire batch to complete
- Progress is visible and engaging

### 2. **Better UX**
- Toast notifications for each completed model
- Live leaderboard updates with smooth animations
- Current model being trained is displayed

### 3. **Fault Tolerance**
- Fallback to polling if SSE connection fails
- Failed models are tracked but don't block progress
- Connection errors are handled gracefully

### 4. **Performance**
- Efficient one-way streaming (server → client)
- Lower overhead than WebSockets for this use case
- Automatic reconnection on network issues

## How It Works

1. **User Starts Training**
   - Frontend sends POST to `/start-training/`
   - Backend responds with task_id
   - Frontend establishes SSE connection to `/training-stream/{task_id}`

2. **Training Process**
   - Backend trains models one by one
   - After each model: `push_model_result(task_id, result)`
   - Result is added to queue

3. **SSE Streaming**
   - SSE endpoint checks queue every second
   - New results are sent as SSE events
   - Frontend receives and updates UI immediately

4. **Completion**
   - When all models done, send completion event
   - Frontend closes SSE connection
   - Shows final results page

## Event Types

### `connected`
```json
{
  "type": "connected",
  "task_id": "abc123"
}
```

### `model_completed`
```json
{
  "type": "model_completed",
  "task_id": "abc123",
  "model": {
    "Model": "Random Forest",
    "R2": 0.8945,
    "RMSE": 0.2341,
    "TT (Sec)": 2.5
  },
  "timestamp": "2025-11-03T10:30:45Z"
}
```

### `completed`
```json
{
  "type": "completed",
  "task_id": "abc123",
  "best_model_name": "Random Forest",
  "total_models_tested": 15,
  "models_saved": 1,
  "leaderboard": [...]
}
```

### `error`
```json
{
  "type": "error",
  "task_id": "abc123",
  "error": "Training failed: insufficient data"
}
```

## Testing

1. **Start Training**
   ```bash
   # Frontend will automatically connect to SSE stream
   ```

2. **Monitor Backend Logs**
   ```
   📊 Pushed model result for task_abc123: Random Forest
   ✅ Completed Linear Regression - Pushed to stream
   ```

3. **Watch Frontend**
   - Leaderboard populates in real-time
   - Toast notifications appear for each model
   - Progress bar advances

## Future Enhancements

1. **Model Training Details**
   - Show hyperparameters being tested
   - Display training/validation curves live
   - Show feature importance as it's calculated

2. **Training Control**
   - Pause/resume training
   - Stop early if satisfactory model found
   - Adjust model list mid-training

3. **Performance Metrics**
   - Show resource usage (CPU/memory)
   - Estimate time remaining
   - Show model training speed

4. **Collaborative Training**
   - Multiple users can watch same training session
   - Share results in real-time
   - Comment on model performance

## Files Modified

### Backend
- `server/ml_training.py` - Added SSE endpoint and streaming logic

### Frontend  
- `client/app/dashboard/blueprints/train/page.tsx` - SSE integration and live UI updates

## Dependencies

### Backend
- `fastapi` - Already installed
- `asyncio` - Python stdlib
- No new dependencies needed!

### Frontend
- `EventSource` API - Native browser API
- No new dependencies needed!

---

**Status**: ✅ Fully Implemented and Ready for Testing
**Impact**: High - Significantly improves user experience during model training
**Complexity**: Medium - Uses standard SSE pattern with fallback
