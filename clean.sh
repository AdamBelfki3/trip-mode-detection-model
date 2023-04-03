target_directory="../trajectories-cleaned"

# only use data directories with labels
rg --files | rg labels.txt | while read -r line ; do
  targ_data=${line:0:9};

  # remove first 6 lines from each data file
  cd $targ_data
    rg --files | rg .plt | while read -r line ; do
      sed -i '' -e '1,6d' $line
    done

  cd ../..
  # concatenate into one large csv
  cat "./${targ_data}Trajectory/"* > "${targ_data}/trajectories.csv"
  # move to clean data dir
  mkdir -p "$target_directory/$targ_data" && cp -r $targ_data "$target_directory/$targ_data";
done
