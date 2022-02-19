# this script is for archiving the logs and models 

mkdir -p archives/
mkdir -p logs/
mkdir -p archives/logs

cp -u logs/* archives/logs -rf
rm logs/ -rf
mkdir -p  logs/


mkdir -p saved_models/
mkdir -p archives/saved_models


cp -u  saved_models/* archives/saved_models -rfn
rm saved_models/ -rf
mkdir -p  saved_models/

if [ "$1" == "bup" ]; then
    cp  * /media/smaran/Storage/_Docs/_Projects/MMED -rf
fi

cat -u lib/model.py>>archives/model_snapshots.py