def read_file(dataset_path):    
    seqs_by_student = {}
    problem_ids = {}
    next_problem_id = 0
    num_lines=0
    with open(dataset_path, 'r') as f:
        for line in f:
            student, problem, is_correct = line.strip().split(' ')
            student = int(student)
            if student not in seqs_by_student:
                seqs_by_student[student] = []
            if problem not in problem_ids:
                problem_ids[problem] = next_problem_id
                next_problem_id += 1
            seqs_by_student[student].append((problem_ids[problem], int(is_correct == '1')))

            num_lines += num_lines
    
    sorted_keys = sorted(seqs_by_student.keys())
    print( " the total numebr of entris in the file are:" + str(num_lines))
    return [seqs_by_student[k] for k in sorted_keys], next_problem_id


def load_dataset(dataset, split_file):
    seqs, num_skills = read_file(dataset)
    
    with open(split_file, 'r') as f:
        student_assignment = f.read().split(' ')
    
    training_seqs = [seqs[i] for i in range(0, len(seqs)) if student_assignment[i] == '1']
    testing_seqs = [seqs[i] for i in range(0, len(seqs)) if student_assignment[i] == '0']
    
    return training_seqs, testing_seqs, num_skills