def compare_code_and_labels(error_file, modified_file):
    with open(error_file, 'r', encoding='utf-8') as f:
        error_lines = f.readlines()
    
    with open(modified_file, 'r', encoding='utf-8') as f:
        modified_lines = f.readlines()

    total_correct_words_modified = 0
    total_lines = len(error_lines)

    for error_line, modified_line in zip(error_lines, modified_lines):

        error_code, labels = error_line.strip().split('|||')
        labels = list(map(int, labels.strip().split()))  
        error_words = error_code.strip().split() 
        

        modified_words = modified_line.strip().split()


        correct_words_modified = 0
        for word, label, modified_word in zip(error_words, labels, modified_words):
            if label == 1 and word != modified_word:
                correct_words_modified += 1
        
        total_correct_words_modified += correct_words_modified

    average_modified = total_correct_words_modified / total_lines if total_lines > 0 else 0
    return average_modified


error_file = 'error_codes.txt'  
modified_file = 'modified_codes.txt'  
average_modified = compare_code_and_labels(error_file, modified_file)
print(f"over_correct: {average_modified:.2f}")
