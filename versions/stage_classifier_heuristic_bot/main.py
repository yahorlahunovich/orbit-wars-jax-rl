import sys
from pathlib import Path

# Add this agent's folder to sys.path so it can find its own 'src' directory
agent_dir = str(Path(__file__).resolve().parent)
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)

from src.bot import agent
