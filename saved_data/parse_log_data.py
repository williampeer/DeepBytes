import sys

log_file_path = sys.stdin.readline("Please enter path to log-file: ")
filename = 'log.txt'  # log_file_path
log_file = file(filename, 'r')

contents = log_file.read()
log_file.close()

# print contents
lines = contents.split('\n')
# print lines
all_data = []
for experiment_ctr in range(len(lines) % 11):

    # sep = lines[11*experiment_ctr].split('x')
    # set_size = int(sep[0][-1])
    # set_parts = int(sep[0][0])

    experiment_data = []
    for i in range(5):
        convergence_line_words = lines[11 * experiment_ctr + 1 + 2 * i].split()
        experiment_data.append(int(convergence_line_words[2]))

        patterns_recalled_line_words = lines[11 * experiment_ctr + 2 + 2 * i].split()
        experiment_data.append(int(patterns_recalled_line_words[1]))
