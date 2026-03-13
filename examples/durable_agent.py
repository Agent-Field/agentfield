import asyncio
import os
import sys
from agentfield import Agent

# Simple agent that counts to 10 with a checkpoint
app = Agent(name="durable-counter")

@app.reasoner("count")
async def counter(input_data, ctx):
    # Check if we are resuming
    execution_id = ctx.execution_id
    print(f"Starting execution: {execution_id}")
    
    saved_state = await app.resume_from_checkpoint(execution_id)
    if saved_state:
        start_count = saved_state.get("count", 0)
        print(f"Resuming from checkpoint! Starting at: {start_count}")
    else:
        start_count = 0
        print("Starting fresh execution.")

    for i in range(start_count, 10):
        print(f"Counting: {i}")
        await asyncio.sleep(1)
        
        # Simulate a checkpoint every 3 steps
        if i > 0 and i % 3 == 0:
            print(f"Creating checkpoint at {i}...")
            await app.checkpoint(state={"count": i + 1}, reason=f"Checkpoint at count {i}")
            
            # To simulate a restart in a demo, we'd normally exit here
            # But for this script, we'll just continue or the user can kill it
            if os.environ.get("SIMULATE_CRASH") == "1":
                print("SIMULATED CRASH!")
                sys.exit(0)

    return {"final_count": 10}

if __name__ == "__main__":
    app.serve()
