import matplotlib.pyplot as plt
import numpy as np


def view():
    data = np.load('training_data20000.npy')
    seq = data[999]
    fig = plt.figure(figsize=(9, 9))

    start_step = 0  # Change to start position if viewing small portion of data
    steps = len(seq)  # Change to smaller constant to limit data view
    rows = len(seq[start_step]) * 2 - 1

    for i in range(steps):
        for j in range(rows):
            if j == 0:
                img = seq[start_step + i][0]
            else:
                img = seq[start_step + i][(j + 1) / 2][np.abs(j % 2 - 1)]
            ax = fig.add_subplot(rows, steps, j * steps + (i + 1))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    plt.show()


def toArrays(data, width, height):
    occ_arrs = []
    vis_arrs = []
    num_seq = data.shape[0]
    num_step = data.shape[1]
    get_vis = data.shape[4] == 2
    for i in range(num_seq):
        occs = []
        viss = []
        for j in range(num_step):
            occ = [[0 for x in range(width)] for y in range(height)]
            if get_vis:
                vis = [[0 for x in range(width)] for y in range(height)]
            for x in range(width):
                for y in range(height):
                    occ[x][y] = data[i, j, x, y, 0]
                    if get_vis:
                        vis[x][y] = data[i, j, x, y, 1]
            occs.append(occ)
            if get_vis:
                viss.append(vis)
        occ_arrs.append(occs)
        if get_vis:
            vis_arrs.append(viss)
    return occ_arrs, vis_arrs


def show_inputs(occ, vis, env, width, height):
    w = width
    h = height
    fig = plt.figure(figsize=(8, 8))
    columns = 10
    rows = 3
    for i in range(1, columns * rows + 1):
        if i <= 10:
            img = env[i - 1]  # env
        elif i <= 20:
            img = occ[i - 11]  # occ
        else:
            img = vis[i - 21]  # vis
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img, cmap='magma')
    plt.show()


def show_outputs(out_arrs, env, width, height):
    w = width
    h = height
    fig = plt.figure(figsize=(8, 8))
    columns = 10
    rows = 2
    for i in range(1, columns * rows + 1):
        if i <= 10:
            img = env[i - 1]  # env
        elif i <= 20:
            img = out_arrs[i - 11]  # occ
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img, cmap='magma')
    plt.show()


def show_predictions(input, output, envs, width, height):
    input_ocs, input_vis = toArrays(input, width, height)

    input_ocs = input_ocs[0]
    input_vis = input_vis[0]
    input_envs = []
    for i in range(10):
        input_envs.append(envs[i, :, :])

    #show_inputs(input_ocs, input_vis, input_envs)

    predicted_arrs, _ = toArrays(output, width, height)
    predicted_arrs = predicted_arrs[0]

    output_env = []
    for i in range(10):
        output_env.append(envs[i + 10, :, :])

    show_outputs(predicted_arrs, output_env, width, height)


def get_stats(output, envs, width, height):
    predicted_arrs, _ = toArrays(output, width, height)
    predicted_arrs = predicted_arrs[0]

    output_env = []
    for i in range(10):
        output_env.append(envs[i + 10, :, :])

    ball_errors = 0
    ball_total = 0
    empty_errors = 0
    empty_total = 0
    for i in range(width):
        for j in range(height):
            if output_env[0][i][j] == 0:   #black
                empty_total += 1
                if predicted_arrs[0][i][j] > 0.5: #predicted closer to white
                    empty_errors += 1
            else:
                ball_total += 1
                if predicted_arrs[0][i][j] < 0.5: #predicted closer to black
                    ball_errors += 1
    ball_error_pct = ball_errors/ball_total
    empty_error_pct = empty_errors/empty_total
    return ball_error_pct,empty_error_pct





def show(data_arr):
    fig = plt.figure()
    plt.imshow(data_arr, cmap='magma', vmin=0, vmax=1, origin='lower')
    plt.show()
