def load_sub_val(sub_val_path='train_sub_val.txt'):
    with open(sub_val_path, mode='r') as f:
        sub_val = set(f.read().splitlines())

    return sub_val