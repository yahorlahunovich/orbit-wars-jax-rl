Introducing "The Producer" agent
I'm sharing my greedy heuristic agent that bases the whole strategy on two assumptions

1) Only send ships when you project it nets you more total production over the next H turns.

2) If you have ships not needed for immediate actions send then to nearby friendly planets that are closer to the enemy.

This simple formulation works surprisingly well. The difficult part was ironing out all the bugs and keeping the world representation in a way that's accurate and efficient. (And eventually discarding all the more complicated ideas)

https://www.kaggle.com/code/slawekbiel/the-producer-agent

Strength: Around 1200 at the moment of sharing Speed: 100-200ms per step