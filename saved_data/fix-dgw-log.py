

log_file = file('dgw-exps-log.txt', 'r')
contents = log_file.read()
log_file.close()

contents = contents.replace("4_x_", "4.")
lines = contents.split('\n')

corrected_contents = ""

for line_ctr in range(len(lines)):
    if line_ctr % 14 != 0:
        corrected_contents += lines[line_ctr] + '\n'

log_f = file('dgw-exps-corrected.txt', 'wb')
log_f.write(corrected_contents)
log_f.close()