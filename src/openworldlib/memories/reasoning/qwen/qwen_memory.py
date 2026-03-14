"""
QwenMemory for multimodal reasoning tasks (e.g., Qwen2.5-Omni)
"""
from ...base_memory import BaseMemory
from typing import Optional, List, Dict, Any


class QwenMemory(BaseMemory):
    """
    Memory module for Qwen multimodal reasoning tasks.
    Stores conversation history including text, images, audios, and videos.
    """
    
    def __init__(self, capacity: int = 100, **kwargs):
        """
        Initialize QwenMemory
        
        Args:
            capacity: Maximum number of conversation turns to store
        """
        super().__init__(capacity=capacity, **kwargs)
        self.storage = []
    
    def record(self, data: Dict[str, Any], **kwargs):
        """
        Record conversation turn to storage
        
        Args:
            data: Dictionary containing:
                - 'messages': List of message dictionaries
                - 'response': Generated response text
                - 'metadata': Optional metadata
        """
        turn_data = {
            'content': data,
            'type': 'conversation',
            'timestamp': len(self.storage),
            'metadata': data.get('metadata', {})
        }
        
        self.storage.append(turn_data)
        
        # Apply capacity limit
        if self.capacity and len(self.storage) > self.capacity:
            self.storage.pop(0)
    
    def select(self, num_turns: int = -1, **kwargs) -> List[Dict]:
        """
        Select recent conversation messages from storage
        
        Args:
            num_turns: Number of recent turns to retrieve (-1 for all)
            
        Returns:
            List of message dictionaries
        """
        if num_turns == -1 or num_turns >= len(self.storage):
            # Return all messages from all stored turns
            all_messages = []
            for turn in self.storage:
                if 'messages' in turn['content']:
                    all_messages.extend(turn['content']['messages'])
                if 'response' in turn['content']:
                    all_messages.append({
                        'role': 'assistant',
                        'content': turn['content']['response']
                    })
            return all_messages
        
        if num_turns <= 0:
            return []
        
        # Return messages from last num_turns
        recent_turns = self.storage[-num_turns:]
        messages = []
        for turn in recent_turns:
            if 'messages' in turn['content']:
                messages.extend(turn['content']['messages'])
            if 'response' in turn['content']:
                messages.append({
                    'role': 'assistant',
                    'content': turn['content']['response']
                })
        return messages
    
    def manage(self, action: str = "reset", **kwargs):
        """
        Manage storage lifecycle
        
        Args:
            action: Management action
                - "reset": Clear all storage
                - "clear_old": Remove oldest turn
        """
        if action == "reset":
            self.storage = []
        elif action == "clear_old" and len(self.storage) > 0:
            self.storage.pop(0)
