
# TCSS 555 - Project
The goal of the "User Profiling in Social Media" project is to build a system for automatic recognition of the age, gender, and personality of Facebook users.




## User Profiling in Social Media
### Images as a source
#### Running script
In order to run the project on images as a source, run the "images.py" file located in the main folder of the repository. Run the script in the following way:

```bash
python images.py -i  "C:\temp\tcss555\public-test-data" -o "C:\temp2\out"
```
where "C:\temp\tcss555\public-test-data" is an example of a directory where public test data is located and "C:\temp2\out" is an example of a directory where the output data will be stored. Change them as needed. File "images.py" also contains a path to the training data ('C:/temp/tcss555/training/'). Change it to the directory where you stored the training data.

#### Code structure
Script running on images as a source uses the following modules contained in the "images-utils" directory:
 * model.py - file containing the architecture of the model built with tensorflow
 * process_csv.py - file containing process_csv, readProfile and age_to_group functions
    * process_csv - processes training csv file creating the dataframe, adding images to it
    * readProfile - opens and reads the csv file
    * age_to_group - given age of a person, retuns an age group it belongs to
* process_data.py - prepares training data and runs it through the model. Trained model is being saved as "age_gender_model.keras"
* process_images.py - processes images using opencv. Images are first grayscaled. Then the faces on the image are cropped. If image doesn't contain a single face, the image is dropped and the user is not considered for the training. If image contains multiple faces, the first face is considered
* process_test_csv.py - file containing process_test_csv and predict functions
    * process_test_csv - processes csv of the test data and calls prediction for each user
    * predict - calls the prediction model "age_gender_model.keras" providing it with the image of a user
* write_users - file containing writeUsers and writeUser functions
    * writeUsers - if there exist any users, this function calls writeUser function for each of them
    * writeUser - creates an xml file with predicted values for the user









