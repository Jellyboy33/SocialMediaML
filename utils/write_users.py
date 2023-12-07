import os

def writeUsers(pred_fp, userIDs): 
    if (not os.path.exists(pred_fp)):
        os.mkdir(pred_fp)
    for index,user in userIDs.iterrows():
        if index > len(userIDs):
            break
        else:
            writeUser(pred_fp,user) 

def writeUser(pred_fp, userIDs): 
    #print(userID)
    userFile = str(userIDs['userid']) + '.xml'
    #change to column name of prediction
    age_group = userIDs['age_group']
    userID = userIDs['userid']
    gender = userIDs['gender']
    open_score = str(3.909)
    con_score = str(3.446)
    ext_score = str(3.487)
    agree_score = str(3.584)
    neur_score = str(2.732)

    output = f'<user id="{userID}"\n' \
             f'age_group="{age_group}"\n' \
             f'gender="{gender}"\n' \
             f'open="{open_score}"\n' \
             f'conscientious="{con_score}"\n' \
             f'extrovert="{ext_score}"\n' \
             f'agreeable="{agree_score}"\n' \
             f'neurotic="{neur_score}"/>'
    
    f = open(os.path.join(pred_fp,userFile),"w")
    f.write(output)
    f.close()