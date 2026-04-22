def process_char_array(char_array):
    processed_values = [abs(float(item.strip('"'))) for item in char_array]
    return processed_values
def compare_files(real_file_path, predicted_file_path):
    try:
        with open(real_file_path, 'r') as real_file, open(predicted_file_path, 'r') as predicted_file:
            real_lines = real_file.readlines()
            predicted_lines = predicted_file.readlines()

            if len(real_lines) != len(predicted_lines):
                raise ValueError("The number of lines in both files does not match.")

            total_action = 0
            correct_action = 0
            total_langth = 0
            correct_langth = 0
            for real_line, predicted_line in zip(real_lines, predicted_lines):
                real_action =[]
                predicted_action =[]
                
                real_words = real_line.strip().split()
                predicted_words = predicted_line.strip().split()
                for i in range(len(real_words)):
                    if real_words[i] =="0":
                        real_action.append(0)
                    elif real_words[i] == "-1":
                         real_action.append(1)
                    elif real_words[i] == "1":
                        real_action.append(2)
                    else:
                        real_action.append(3)
                for i in range(len(predicted_words)):
                    if predicted_words[i] =="0":
                        predicted_action.append(0)
                    elif predicted_words[i] == "-1":
                         predicted_action.append(1)
                    elif predicted_words[i] == "1":
                        predicted_action.append(2)
                    else:
                        predicted_action.append(3)
                for real_word, predicted_word in zip(real_action, predicted_action):
                    total_action += 1
                    if real_word == predicted_word:
                        correct_action += 1
                real_action = process_char_array(real_words)
                predicted_action = process_char_array(predicted_words)
                for real_word, predicted_word in zip(real_action, predicted_action):
                    total_langth += 1
                    if real_word == predicted_word:
                        correct_langth += 1

            accuracy_action = correct_action / total_action * 100
            print(f"Repair Action Accuracy: {accuracy_action:.2f}%")
            accuracy_length = correct_langth / total_langth * 100
            print(f"Repair Length Accuracy: {accuracy_length:.2f}%")
    
    except Exception as e:
        print(f"An error occurred: {e}")


real_file_path = 'real_data.txt'       
predicted_file_path = 'predicted_data.txt'  

compare_files(real_file_path, predicted_file_path)
