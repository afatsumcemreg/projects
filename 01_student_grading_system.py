"""
In this project, you will create a student grading system. What is required of you:
* Choose a lesson for yourself.  (Mathematics, Physics, Linear Algebra etc.)
* Create your note range (100-80 ⇒ A, 79-70 ⇒ B etc.)
* Create a system where you can enter Student Information (Name, Surname, School Number, exam score) and keep this information.
* Based on the information entered, the student must show whether or not he has passed the course.
* If the student has passed the course, the text "Passed" should be shown in the field where the student's information is kept, and "Failed" if the student did not pass the course.
* Create a Dataframe that shows the students whose grades have been entered, who passed the course and those who did not.
* Create a DataframeConvert the created Dataframe to an Excel table.
"""

# Importing libraries
import pandas as pd

# Create emypty list
name_surname = []
school_number = []
exam_grade = []
letter_grade = []
situation = []

# input data
new_name_surname = input("Enter name and surname: ")
name_surname.append(new_name_surname)
new_school_number = input("Enter school number: ")
school_number.append(new_school_number)
new_exam_grade = float(input("Enter exam grade: "))
exam_grade.append(new_exam_grade)

# Enter the conditions for the successful situation
if 90 <= new_exam_grade <= 100:
    letter_grade.append("AA")
    situation.append("Successful")
elif 85 <= new_exam_grade <= 89:
    letter_grade.append("BA")
    situation.append("Successful")
elif 80 <= new_exam_grade <= 84:
    letter_grade.append("BB")
    situation.append("Successful")
elif 75 <= new_exam_grade <= 79:
    letter_grade.append("CB")
    situation.append("Successful")
elif 60 <= new_exam_grade <= 74:
    letter_grade.append("CC")
    situation.append("Successful")
elif 0 <= new_exam_grade <= 59:
    letter_grade.append("FF")
    situation.append("Unsuccessful")
elif new_exam_grade > 100:
    print("You entered wrong information. Try again.")

# Situation detection regarding successful
df = pd.DataFrame (situation, name_surname, columns=["Situation"])
df["School Number"] = school_number
df["Letter Grade"] = letter_grade
df

# Converting the dataframe to an exell table
file = pd.ExcelWriter("grade_situation.xlsx")
df.to_excel(file)
file.save()