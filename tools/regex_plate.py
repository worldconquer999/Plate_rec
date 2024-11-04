import re

def process_true_plate(recognizer_result: str):
    recognizer_result = list(recognizer_result)
    if recognizer_result[0] in ["D", "Q"]:
        recognizer_result[0] = "0"
    if recognizer_result[0] in ["B"]:
        recognizer_result[0] = "8"
    if recognizer_result[1] in ["D", "Q"]:
        recognizer_result[1] = "0"
    if recognizer_result[1] in ["B"]:
        recognizer_result[1] = "8"
    
    if len(recognizer_result) > 3:
        for index, char in enumerate(recognizer_result[3:]):
            if char in ["D", "Q"]: recognizer_result[3 + index] = "0"
            if char in ["B"]: recognizer_result[3 + index] = "8"

    plate = "".join(recognizer_result)
    return plate.replace("M0", "MD")

if __name__ == "__main__":
    test_cases = [
        ["3DA18786", "30A18786"],
        ["3QA18786", "30A18786"],
        ["30A1B786", "30A18786"],
        ["30A1878D", "30A18780"],
        ["30A18Q86", "30A18086"],
        ["30M01234", "30MD1234"]
    ]
    num_true = 0
    for case in test_cases:
        plate = process_true_plate(case[0])
        if plate == case[1]:
            num_true += 1
        else:
            print(case, plate)
    
    print(num_true / len(test_cases))