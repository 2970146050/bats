# 先自动创建输出文件夹（必须加）
mkdir -p eval/batch
mkdir -p eval/cola
mkdir -p eval/lamba

# ==================== batch ====================
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.0best.pt > eval/batch/0.0.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.1best.pt > eval/batch/0.1.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.2best.pt > eval/batch/0.2.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.3best.pt > eval/batch/0.3.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.4best.pt > eval/batch/0.4.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.5best.pt > eval/batch/0.5.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.6best.pt > eval/batch/0.6.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.7best.pt > eval/batch/0.7.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch0.8best.pt > eval/batch/0.8.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/batch-m/batch1.0best.pt > eval/batch/1.0.txt 2>&1

# ==================== cola ====================
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola1.0best.pt > eval/cola/1.0.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola0.9best.pt > eval/cola/0.9.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola0.8best.pt > eval/cola/0.8.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola0.7best.pt > eval/cola/0.7.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola0.6best.pt > eval/cola/0.6.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola0.4best.pt > eval/cola/0.4.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola0.3best.pt > eval/cola/0.3.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola0.2best.pt > eval/cola/0.2.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/cola/cola0.1best.pt > eval/cola/0.1.txt 2>&1

# ==================== lamba ====================
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.0best.pt > eval/lamba/0.0.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.1best.pt > eval/lamba/0.1.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.2best.pt > eval/lamba/0.2.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.3best.pt > eval/lamba/0.3.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.4best.pt > eval/lamba/0.4.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.5best.pt > eval/lamba/0.5.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.6best.pt > eval/lamba/0.6.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.7best.pt > eval/lamba/0.7.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.8best.pt > eval/lamba/0.8.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba0.9best.pt > eval/lamba/0.9.txt 2>&1
python eval_all_cifar10.py --model-path /home/tgj/project/TDAT-origin/TDAT-origin/trainedModels/lamba/lamba1.0best.pt > eval/lamba/1.0.txt 2>&1