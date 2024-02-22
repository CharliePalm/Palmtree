
## Palmtree

Palmtree is an attempt at making monte-carlo reinforcement learning less strenuous and more budget-hardware friendly in addition to comparing how studying GM positions compares to MCTS.

# Usage

  git clone https://github.com/CharliePalm/Palmtree
  pip install -r requirements.txt
  python3 driver.py -t -e 100

# Hardware

Various tests were conducted on Nvidia GTX 1060 (3gb VRAM), Radeon RX 580 (8gb VRAM), and Radeon RX 6800 (16gb VRAM).
The primary discovery here is how difficult Radeon's tensorflow and pytorch integrations are to use. 

The options are either:
 - use linux and ROCM and suffer from numerous bugs related to memory management
 - use PLAIDML which requires an older version of tensorflow which will throw sporadic errors related to imports

Neither are particularly attractive, and led to quite a few headaches. Of the tested cards, the 6800 is by far the highest performer, though with the amount of debugging radeon made me do it doesn't feel like it.

It's also worth noting that the main branch currently is using plaidml. If I decide to resurrect this project in the future, should ROCM dramatically improve, I'll add more dynamic options for software usage.

For NVIDIA, using the 1060 was easy but obviously not particularly performative. To get a model that could play for more than 5 moves without imploding I would need to train for several days. This isn't surprising, I just wish I had access to more high end NVIDIA cards to train with.

# Findings
In the end, the primary difference between MCTS and pure 'image classification' type models was the ~flavor~ of play. 
This is to say, the depth of the MCTS search was limited for hardware purposes, so it never learned openings particularly well, while openings made up the majority of the positions the image classification model saw, so it frequently played classical chess theory up until about move 3 or 4.
However, since the image classification had no lookahead, once its opponent blundered a piece, it rarely saw how to take advantage of it, as it rarely saw blunders of that caliber in its learning.

This was the primary advantage of the MCTS model - once it saw something promising it usually knew how to take advantage of it. One drawback of this, however, is that the MCTS model would often get overexcited if it saw a checkmate in its lookahead, either for or against it. 
This resulted in it always beating the image-classification model, almost always losing to humans, and almost always drawing with itself, as it would just play defensively forever.
