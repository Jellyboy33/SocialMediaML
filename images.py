import sys
sys.path.append('./images-utils')
from process_csv import process_csv
from process_test_csv import process_test_csv

def main():
    args = sys.argv[1:]
   
    #obtain filepaths for input and output
    test_data_fp = args[1]
    pred_fp = args[3]

    #///CHANGE TO CORRECT TRAINING FILEPATH///
    train_data_fp = 'C:/temp/tcss555/training/'

    # test_profile_df = readProfile(test_data_fp)

    process_csv(train_data_fp)
    process_test_csv(test_data_fp, pred_fp)

if __name__ == "__main__":
    main()