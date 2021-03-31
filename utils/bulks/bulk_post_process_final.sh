source ~/venv/pytorch/bin/activate
ps=(050 080 100)
ps=(050)
# annt_types=(top all)
annt_types=(all)
# tag_types=(tag notag)
tag_types=(notag)
# inln_types=(sinln stinln none)
inln_types=(stinln)

ks=(0 1 2 3 4 5)
# tasks=(task1_all task2_keyword_set1 task2_keyword_set2 task2_keyword_set3 task2_keyword_set_all)
# tasks=(task1_all task2_keyword_set_all)
tasks=(task1_certain0 task2_keyword_set_certain0)
# tasks=(task1_certain0 task1_certain1 task1_certain2 task2_keyword_set_certain0 task2_keyword_set_certain1 task2_keyword_set_certain2)

for p in ${ps[@]}; do
  for annt_type in ${annt_types[@]}; do
    for inln_type in ${inln_types[@]}; do
      # if [ $inln_type != none ]; then
      #   python ../post_process.py -d jijinews  -y final/css_prob${p}_${annt_type}_${inln_type}/base/org.yml
      # fi
      # python ../post_process.py -d jijinews  -y final/css_prob${p}_${annt_type}_${inln_type}/con/org.yml
      # python ../post_process.py -d jijinews  -y final/css_prob${p}_${annt_type}_${inln_type}/sum/org.yml

      for task in ${tasks[@]}; do
        if [ $inln_type != none ]; then
          # python ../post_process.py -d jijinews  -y final/css_prob${p}_${annt_type}_${inln_type}/base/${task}.yml
          python ../post_process.py -d jijinews  -y final/css_prob${p}_${annt_type}_${inln_type}/base/${task}_10best.yml
        fi
        # python ../post_process.py -d jijinews  -y final/css_prob${p}_${annt_type}_${inln_type}/con/${task}.yml
        # python ../post_process.py -d jijinews  -y final/css_prob${p}_${annt_type}_${inln_type}/sum/${task}.yml
      done
    done
  done
done
for tag_type in ${tag_types[@]}; do
  for inln_type in ${inln_types[@]}; do
    # if [ $inln_type != none ]; then
    #   python ../post_process.py -d jijinews  -y final/css_synthetic_all_${tag_type}_${inln_type}/base/org.yml
    # fi
    # python ../post_process.py -d jijinews  -y final/css_synthetic_all_${tag_type}_${inln_type}/con/org.yml
    # python ../post_process.py -d jijinews  -y final/css_synthetic_all_${tag_type}_${inln_type}/sum/org.yml

    for task in ${tasks[@]}; do
      if [ $inln_type != none ]; then
        # python ../post_process.py -d jijinews  -y final/css_synthetic_all_${tag_type}_${inln_type}/base/${task}.yml
        # python ../post_process.py -d jijinews  -y final/css_synthetic_all_${tag_type}_${inln_type}/base/${task}_10best.yml
      fi
      # python ../post_process.py -d jijinews  -y final/css_synthetic_all_${tag_type}_${inln_type}/con/${task}.yml
      # python ../post_process.py -d jijinews  -y final/css_synthetic_all_${tag_type}_${inln_type}/sum/${task}.yml
    done
  done
done

for task in ${tasks[@]}; do
  # python ../post_process.py -d jijinews -y css_srcannt010/base/test_baobab_${task}.yml
  # python ../post_process.py -d jijinews -y css_srcannt010/base/test_baobab_${task}_10best.yml
done
