for rare_mode in 'drop'
do
    for lang in 'ger_n' 'cze'
    do
        echo 'Parse' $lang
        python src/h01_data/parse.py --lang $lang --rare-mode $rare_mode
    done
done