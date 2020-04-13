run_type=''
is_opt=''
langs='ger_n cze'
for model in 'lstm'
do
    for context in 'none' 'word2vec'
    do
        echo $model
        python src/h02_learn/train${run_type}.py --model $model $is_opt --languages $langs --context $context
    done
done

run_type='_bayes'
is_opt=''
for model in 'lstm'
do
    for context in 'none' 'word2vec'
    do
        echo $model
        python src/h02_learn/train${run_type}.py --model $model $is_opt --languages $langs --context $context
    done
done

run_type='_cv'
is_opt='--opt'
for model in 'lstm'
do
    for context in 'none' 'word2vec'
    do
        echo $model
        python src/h02_learn/train${run_type}.py --model $model $is_opt --languages $langs --context $context
    done
done


run_type=''
is_opt=''
context='word2vec'
langs='ger_n cze'
for model in 'mlp-word2vec'
do
    echo $model
    python src/h02_learn/train${run_type}.py --model $model $is_opt --languages $langs
done

run_type='_bayes'
is_opt=''
for model in 'mlp-word2vec'
do
    echo $model
    python src/h02_learn/train${run_type}.py --model $model $is_opt --languages $langs
done

run_type='_cv'
is_opt='--opt'
for model in 'mlp-word2vec'
do
    echo $model
    python src/h02_learn/train${run_type}.py --model $model $is_opt --languages $langs
done
