# fast_style_transfer

# Produce results on floydhub

Modify the floyd_train.py file - set content image, style image and other parameters
and the run one of the following commands: 

* Run on CPU: floyd run --cpu --data monoton/datasets/ai_doodle/3:/data --env keras "python floyd_train.py"

* Run on GPU: floyd run --gpu --data monoton/datasets/ai_doodle/3:/data --env keras "python floyd_train.py"