"""
LiteLLM Async Task Cleanup Patch

This module patches LiteLLM's asynchronous logging to ensure all tasks complete properly
and prevent "Task was destroyed but it is pending!" errors.
"""

import asyncio
import logging
import functools
import inspect
import sys
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Global registry of pending LiteLLM async tasks
_pending_tasks = set()

def _patch_litellm_async_logging():
    """
    Patches LiteLLM async logging functions to ensure proper task cleanup.
    Prevents "Task was destroyed but it is pending!" errors.
    """
    try:
        # Try to import LiteLLM modules
        import litellm
        from litellm.utils import _client_async_logging_helper
        
        # Store original function
        original_client_async_logging = _client_async_logging_helper
        
        # Create patched version with error handling
        @functools.wraps(original_client_async_logging)
        async def patched_client_async_logging_helper(*args, **kwargs):
            try:
                return await original_client_async_logging(*args, **kwargs)
            except Exception as e:
                logger.warning(f"LiteLLM async logging error (handled): {e}")
                return None
                
        # Apply patch
        litellm.utils._client_async_logging_helper = patched_client_async_logging_helper
        
        # Patch Logging class async_success_handler if available
        if hasattr(litellm, 'litellm_core_utils') and hasattr(litellm.litellm_core_utils, 'litellm_logging'):
            from litellm.litellm_core_utils.litellm_logging import Logging
            
            if hasattr(Logging, 'async_success_handler'):
                original_async_success_handler = Logging.async_success_handler
                
                @functools.wraps(original_async_success_handler)
                async def patched_async_success_handler(*args, **kwargs):
                    try:
                        return await original_async_success_handler(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"LiteLLM async_success_handler error (handled): {e}")
                        return None
                        
                Logging.async_success_handler = patched_async_success_handler
        
        logger.info("Successfully patched LiteLLM async logging functions")
        return True
        
    except ImportError:
        logger.warning("Could not find LiteLLM modules to patch")
        return False
    except Exception as e:
        logger.error(f"Error patching LiteLLM: {e}")
        return False


def create_task_with_cleanup(coro: Coroutine) -> asyncio.Task:
    """
    Creates an asyncio task with automatic cleanup registration.
    Prevents orphaned tasks and associated warnings.
    """
    task = asyncio.create_task(coro)
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
    return task


async def cleanup_all_async_tasks(timeout: float = 2.0):
    """
    Waits for pending async tasks to complete within timeout period.
    Should be called before exiting async contexts to prevent warnings.
    """
    if not _pending_tasks:
        return
    
    logger.debug(f"Cleaning up {len(_pending_tasks)} pending async tasks...")
    try:
        # Wait for all pending tasks with a timeout
        done, pending = await asyncio.wait(
            _pending_tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
        )
        
        if pending:
            logger.warning(f"{len(pending)} async tasks still pending after timeout")
    except Exception as e:
        logger.error(f"Error during async task cleanup: {e}")


# Apply the patch when this module is imported
_patch_litellm_async_logging()