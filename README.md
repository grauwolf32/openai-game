### Prerequisites

```
tensorflow
pygame
gym # https://github.com/openai/gym/
baselines # https://github.com/openai/baselines/
```

### How to use

To start learn new model
```
python3 run_acktr.py 

OPTIONS:
--fname - path to checkpoint file
--env - name of the environment (gathering | pursuit) 
```

To run game simulation:
```
python3 ai_runner.py

OPTIONS:
--fname - path to checkpoint file (required)
--env - name of the environment (gathering by default)
```

