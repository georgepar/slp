
RES_DIR=$1

echo -e "MAE:\t\t\t\c"
cat ${RES_DIR}/* | grep mae | awk '{ sum += $2; n++  } END { if (n > 0) print sum / n;  }'
echo -e "Correlation:\t\t\c"
cat ${RES_DIR}/* | grep corr | awk '{ sum += $2; n++  } END { if (n > 0) print sum / n;  }'
echo -e "5-class Accuracy:\t\c"
cat ${RES_DIR}/* | grep acc_5 | awk '{ sum += $2; n++  } END { if (n > 0) print sum / n;  }'
echo -e "7-class Accuracy:\t\c"
cat ${RES_DIR}/* | grep acc_7 | awk '{ sum += $2; n++  } END { if (n > 0) print sum / n;  }'
echo -e "F1 Score:\t\t\c"
cat ${RES_DIR}/* | grep f1 | awk '{ sum += $2; n++  } END { if (n > 0) print sum / n;  }'
echo -e "Binary Accuracy:\t\c"
cat ${RES_DIR}/* | grep bin_acc | awk '{ sum += $2; n++  } END { if (n > 0) print sum / n;  }'
