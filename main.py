import sys


print("SUPERVISED LEARNING OF PROCEDURES FROM TUTORIAL VIDEOS")

print("\n")

if sys.argv[1]=="a":
    import academic_main as am
    am.academic(sys.argv[2])
elif sys.argv[1]=="e":
    import exercise_main as em
    em.exercise(sys.argv[2])
elif sys.argv[1]=="webcam":
    import exercise_main as em
    em.exercise('webcam')
else:
    print("Wrong option ")

