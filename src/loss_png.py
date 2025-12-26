import matplotlib.pyplot as plt

file_path = "outputs/stage1/train3/log.txt"
with open(file_path, "r") as f:
    lines = f.readlines()

loss_values = []
idx = 1
start_read = False
for line in lines:
    if f"I Epoch {idx}/300" in line:
        start_read = True
    if start_read and "I loss/online/total/E1:" in line:
        start_read = False
        idx += 1
        parts = line.split("I loss/online/total/E1:")
        if len(parts) > 1:
            loss_str = parts[1].strip().split()[0]
            try:
                loss_value = float(loss_str)
                loss_values.append(loss_value)
            except ValueError:
                continue
print(f"Extracted {len(loss_values)} loss values.")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_values) + 1), loss_values)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
