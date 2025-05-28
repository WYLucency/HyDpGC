#
for method in hdgc; do
  for dataset in cora flickr ogbn-arxiv; do
    case $dataset in
      cora)
        for r in 0.5; do
          for e in 10; do
              python run_eval.py -M $method -D $dataset -R $r --target_epsilon $e --down_epsilon $e
          done
        done
        ;;
      ogbn-arxiv)
        for r in 0.001 0.005 0.01; do
          for e in 10; do
            python run_eval.py -M $method -D $dataset -R $r --target_epsilon $e --down_epsilon $e
          done
        done
        ;;
      flickr)
        for r in 0.001 0.005 0.01; do
          for e in 10; do
            python run_eval.py -M $method -D $dataset -R $r --target_epsilon $e --down_epsilon $e
          done
        done
        ;;
    esac
  done
done
