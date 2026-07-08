from core.pipeline.context import PipelineContext
from core.pipeline.stage import StageResult
from core.domain.story.bible import StoryBible

class ContextReducer:
    """
    Pure Redux-style reducer that takes a frozen PipelineContext and a validated StageResult,
    and returns a brand new PipelineContext (Context N+1) preserving identity through Pydantic model_copy.
    Does not run validations or retries.
    """
    def reduce(self, context, result: StageResult):
        from core.pipeline.compiler_context import CompilerContext
        
        is_compiler_context = isinstance(context, CompilerContext)
        pipeline_context = context.pipeline if is_compiler_context else context
        
        updates = {}
        
        # Merge StoryBible
        if isinstance(result.artifact, StoryBible):
            updates['story_bible'] = result.artifact
            
        # Merge ExecutionNode (The true canonical runtime output)
        if hasattr(result, 'execution_node') and result.execution_node:
            updates['execution_nodes'] = pipeline_context.execution_nodes + [result.execution_node]
            
        # Store other outputs in ephemeral state
        new_state = dict(pipeline_context.state)
        new_state[str(type(result.artifact).__name__)] = result.artifact
        updates['state'] = new_state
        
        new_pipeline = pipeline_context.model_copy(update=updates)
        
        if is_compiler_context:
            return CompilerContext(new_pipeline, context.workspace, context.registry, context.queue)
        return new_pipeline
