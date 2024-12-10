import os
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from intvalpy import IntLinIncR2 as IntLinIncR2_original, Interval, Tol, precision

# Включение расширенной точности
precision.extendedPrecisionQ = True

DATA_PATH = 'data'

def load_data(directory, side):
    values_x = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    num_sides = 8
    num_samples = 1024
    num_values = len(values_x)
    loaded_data = np.full((num_sides, num_samples, num_values * 100, 2), 0.0)
    
    # Заполнение координат X
    for i in range(num_sides):
        for j in range(num_samples):
            loaded_data[i][j, :, 0] = np.repeat(values_x, 100)
    
    # Загрузка данных из файлов
    for offset, value_x in enumerate(values_x):
        # Форматирование имени файла без лишних десятичных точек
        if value_x == 0:
            value_x_str = "0"
        elif value_x.is_integer():
            value_x_str = f"{int(value_x)}"
        else:
            value_x_str = f"{value_x:g}"  # Убирает лишние нули
        
        file_name = f"{value_x_str}lvl_side_{side}_fast_data.json"
        file_path = os.path.join(directory, file_name)
        
        if not os.path.isfile(file_path):
            print(f"Внимание: Файл {file_name} не найден. Пропускаем.")
            continue
        
        with open(file_path, "rt") as f:
            data = json.load(f)
        
        for i in range(num_sides):
            for j in range(num_samples):
                sensor_data = data["sensors"][i][j]
                loaded_data[i][j, offset * 100:offset * 100 + len(sensor_data), 1] = sensor_data
    
    return loaded_data

def regression_type_1(points):
    x, y = points[:, 0], points[:, 1]
    weights = np.full(len(y), 1 / 16384)
    
    # Построение Interval объектов
    X_mat = Interval([[[x_el, x_el], [1, 1]] for x_el in x])
    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    
    # Максимизация Tol
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)
    updated = 0
    
    if tol_val < 0:
        for i in range(len(Y_vec)):
            X_mat_small = Interval([[[x[i], x[i]], [1, 1]]])
            Y_vec_small = Interval([[y[i], weights[i]]], midRadQ=True)
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                weights[i] = abs(y[i] - (x[i] * b_vec[0] + b_vec[1])) + 1e-8
                updated += 1
    
    Y_vec = Interval([[y_el, weights[i]] for i, y_el in enumerate(y)], midRadQ=True)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat, Y_vec)
    
    return b_vec, weights, updated

def regression_type_2(points):
    x, y = points[:, 0], points[:, 1]
    eps = 1 / 16384
    
    x_new = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    y_ex_up = np.full(11, -np.inf)
    y_ex_down = np.full(11, np.inf)
    y_in_up = np.full(11, -np.inf)
    y_in_down = np.full(11, np.inf)
    
    for i in range(len(x_new)):
        y_list = np.sort(y[i * 100 : (i + 1) * 100])
        if len(y_list) < 100:
            print(f"Предупреждение: Недостаточно данных для x={x_new[i]}.")
            continue
        y_in_down[i] = y_list[24] - eps
        y_in_up[i] = y_list[74] + eps
        y_ex_up[i] = min(y_list[74] + 1.5 * (y_list[74] - y_list[24]), y_list[-1])
        y_ex_down[i] = max(y_list[24] - 1.5 * (y_list[74] - y_list[24]), y_list[0])
    
    X_mat = []
    Y_vec = []
    for i in range(len(x_new)):
        x_el = x_new[i]
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_ex_up[i]])
        
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_ex_down[i], y_in_up[i]])
        
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_ex_up[i]])
        
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])
    
    X_mat = np.array(X_mat)
    Y_vec = np.array(Y_vec)
    
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)
    to_remove = []
    
    if tol_val < 0:
        for i in range(len(Y_vec)):
            X_mat_small = Interval([X_mat[i]])
            Y_vec_small = Interval([Y_vec[i]], midRadQ=True)
            value = Tol.value(X_mat_small, Y_vec_small, b_vec)
            if value < 0:
                to_remove.append(i)
    
        X_mat = np.delete(X_mat, to_remove, axis=0)
        Y_vec = np.delete(Y_vec, to_remove, axis=0)
    
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)
    
    vertices1 = IntLinIncR2_fix(A=X_mat_interval, b=Y_vec_interval)
    vertices2 = IntLinIncR2_fix(A=X_mat_interval, b=Y_vec_interval, consistency='tol')
    
    plt.xlabel("b0")
    plt.ylabel("b1")
    
    b_uni_vertices = []
    for v in vertices1:
        if len(v) > 0:
            x_v, y_v = v[:, 0], v[:, 1]
            b_uni_vertices.extend([(x_v[i], y_v[i]) for i in range(len(x_v))])
            plt.fill(x_v, y_v, linestyle='-', linewidth=1, color='gray', alpha=0.5, label="Uni")
            plt.scatter(x_v, y_v, s=0, color='black', alpha=1)
    
    b_tol_vertices = []
    for v in vertices2:
        if len(v) > 0:
            x_v, y_v = v[:, 0], v[:, 1]
            b_tol_vertices.extend([(x_v[i], y_v[i]) for i in range(len(x_v))])
            plt.fill(x_v, y_v, linestyle='-', linewidth=1, color='blue', alpha=0.3, label="Tol")
            plt.scatter(x_v, y_v, s=10, color='black', alpha=1)
    
    plt.scatter([b_vec[0]], [b_vec[1]], s=10, color='red', alpha=1, label="argmax Tol")
    plt.legend()
    return b_vec, (y_in_down, y_in_up), (y_ex_down, y_ex_up), to_remove, b_uni_vertices, b_tol_vertices

def build_plots(data, coord_x, coord_y):
    # Метод 1
    b_vec, rads, to_remove = regression_type_1(data)
    x, y = data[:, 0], data[:, 1]
    
    plt.figure()
    plt.title(f"Y(x) метод 1 для ({coord_x}, {coord_y})")
    plt.scatter(x, y, label="медианы")
    plt.plot([-0.5, 0.5], [b_vec[1] + b_vec[0] * -0.5, b_vec[1] + b_vec[0] * 0.5], label="Argmax Tol")
    plt.legend()
    
    plt.figure()
    plt.title(f"Y(x) - b0*x - b1 метод 1 для ({coord_x}, {coord_y})")
    for i in range(len(y)):
        plt.plot([i, i], [y[i] - rads[i] - b_vec[1] - b_vec[0] * x[i],
                          y[i] + rads[i] - b_vec[1] - b_vec[0] * x[i]], color="k", zorder=1)
        plt.plot([i, i], [y[i] - 1 / 16384 - b_vec[1] - b_vec[0] * x[i],
                          y[i] + 1 / 16384 - b_vec[1] - b_vec[0] * x[i]], color="blue", zorder=2)
    
    # Метод 2
    plt.figure()
    plt.title(f"Uni и Tol метод 2 для ({coord_x}, {coord_y})")
    b_vec2, y_in, y_ex, to_remove, b_uni_vertices, b_tol_vertices = regression_type_2(data)
    print(f"({coord_x}, {coord_y}) Метод 2: b0={b_vec2[0]}, b1={b_vec2[1]}, Удалено={len(to_remove)}")
    
    x2 = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    for i in range(len(x2)):
        plt.plot([x2[i], x2[i]], [y_ex[0][i], y_ex[1][i]], color="gray", zorder=1)
        plt.plot([x2[i], x2[i]], [y_in[0][i], y_in[1][i]], color="blue", zorder=2)
    
    plt.plot([-0.5, 0.5], [b_vec2[1] + b_vec2[0] * -0.5, b_vec2[1] + b_vec2[0] * 0.5],
             label="Argmax Tol", color="red", zorder=1000)
    
    # Проверяем, не пусты ли b_uni_vertices и b_tol_vertices
    if not b_uni_vertices:
        print("Предупреждение: b_uni_vertices пуст, пропускаем построение полигонов для Uni.")
    if not b_tol_vertices:
        print("Предупреждение: b_tol_vertices пуст, пропускаем построение полигонов для Tol.")
    
    # Строим заполненные области, только если есть вершины
    if b_uni_vertices and b_tol_vertices:
        x2_extended = np.concatenate(([-3], x2, [3]))
        
        # Отображаем область для Uni
        for i in range(len(x2_extended) - 1):
            x0, x1 = x2_extended[i], x2_extended[i + 1]
            
            # Если список вершин пуст, дальше идти нет смысла
            if not b_uni_vertices:
                break
            
            # Ищем min и max для b_uni_vertices
            max_val = b_uni_vertices[0][1] + b_uni_vertices[0][0] * (x0 + x1) / 2
            min_val = max_val
            max_idx, min_idx = 0, 0
        
            for j in range(len(b_uni_vertices)):
                val = b_uni_vertices[j][1] + b_uni_vertices[j][0] * (x0 + x1) / 2
                if max_val < val:
                    max_idx, max_val = j, val
                if min_val > val:
                    min_idx, min_val = j, val
        
            y0_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x0
            y1_low = b_uni_vertices[min_idx][1] + b_uni_vertices[min_idx][0] * x1
            y0_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x0
            y1_hi = b_uni_vertices[max_idx][1] + b_uni_vertices[max_idx][0] * x1
            plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi],
                     facecolor="lightgray", linewidth=0)
        
        # Отображаем область для Tol
        for i in range(len(x2_extended) - 1):
            x0, x1 = x2_extended[i], x2_extended[i + 1]
            
            if not b_tol_vertices:
                break
            
            max_val = b_tol_vertices[0][1] + b_tol_vertices[0][0] * (x0 + x1) / 2
            min_val = max_val
            max_idx, min_idx = 0, 0
        
            for j in range(len(b_tol_vertices)):
                val = b_tol_vertices[j][1] + b_tol_vertices[j][0] * (x0 + x1) / 2
                if max_val < val:
                    max_idx, max_val = j, val
                if min_val > val:
                    min_idx, min_val = j, val
        
            y0_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x0
            y1_low = b_tol_vertices[min_idx][1] + b_tol_vertices[min_idx][0] * x1
            y0_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x0
            y1_hi = b_tol_vertices[max_idx][1] + b_tol_vertices[max_idx][0] * x1
            plt.fill([x0, x1, x1, x0], [y0_low, y1_low, y1_hi, y0_hi],
                     facecolor="lightblue", linewidth=0)
    
    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.6, 0.6))

def amount_of_neg(all_data, coord_x, coord_y):
    points = all_data[coord_y][coord_x]
    x, y = points[:, 0], points[:, 1]
    eps = 1 / 16384
    
    x_new = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    y_ex_up = np.full(11, -np.inf)
    y_ex_down = np.full(11, np.inf)
    y_in_up = np.full(11, -np.inf)
    y_in_down = np.full(11, np.inf)
    
    for i in range(len(x_new)):
        y_list = np.sort(y[i * 100 : (i + 1) * 100])
        if len(y_list) < 100:
            continue
        y_in_down[i] = y_list[24] - eps
        y_in_up[i] = y_list[74] + eps
        y_ex_up[i] = min(y_list[74] + 1.5 * (y_list[74] - y_list[24]), y_list[-1])
        y_ex_down[i] = max(y_list[24] - 1.5 * (y_list[74] - y_list[24]), y_list[0])
    
    X_mat = []
    Y_vec = []
    for i in range(len(x_new)):
        x_el = x_new[i]
        X_mat.append([[x_el, x_el], [1, 1]])
        Y_vec.append([y_in_down[i], y_in_up[i]])
    
    X_mat = np.array(X_mat)
    Y_vec = np.array(Y_vec)
    
    X_mat_interval = Interval(X_mat)
    Y_vec_interval = Interval(Y_vec)
    to_remove = []
    b_vec, tol_val, num_iter, calcfg_num, exit_code = Tol.maximize(X_mat_interval, Y_vec_interval)
    
    for i in range(len(Y_vec)):
        X_mat_small = Interval([X_mat[i]])
        Y_vec_small = Interval([Y_vec[i]], midRadQ=True)
        value = Tol.value(X_mat_small, Y_vec_small, b_vec)
        if value < 0:
            to_remove.append(i)
    
    return len(to_remove)

def unique(a, decimals=12):
    if a.size == 0:
        return a
    a = np.ascontiguousarray(a)
    a = np.round(a, decimals=int(decimals))
    dtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    _, idx = np.unique(a.view(dtype), return_index=True)
    idx = np.sort(idx)
    return a[idx]

def clear_zero_rows(a, b, ndim=2):
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    a = np.round(a, decimals=12)
    b = np.round(b, decimals=12)
    
    condition = (np.sum(np.abs(a) <= 1e-12, axis=1) == ndim) & (b > 0)
    cnmty = not np.any(condition)
    
    index = np.where(np.sum(np.abs(a) <= 1e-12, axis=1) != ndim)[0]
    return a[index], b[index], cnmty

def BoundaryIntervals(A, b):
    m, n = A.shape
    S = []
    
    for i in range(m):
        q = [-np.inf, np.inf]
        si = True
        try:
            dotx = (A[i] * b[i]) / np.dot(A[i], A[i])
        except ZeroDivisionError:
            print(f"Ошибка: Нулевой знаменатель при вычислении dotx для строки {i}. Пропускаем.")
            continue
        
        p = np.array([-A[i, 1], A[i, 0]])
        
        for k in range(m):
            if k == i:
                continue
            Akx = np.dot(A[k], dotx)
            c = np.dot(A[k], p)
            
            if np.sign(c) == -1:
                tmp = (b[k] - Akx) / c
                q[1] = min(q[1], tmp)
            elif np.sign(c) == 1:
                tmp = (b[k] - Akx) / c
                q[0] = max(q[0], tmp)
            else:
                if Akx < b[k]:
                    if np.dot(A[k], A[i]) > 0:
                        si = False
                        break
                    else:
                        return np.array([])
        
        if q[0] > q[1]:
            si = False
        
        p = p + 1e-301  # Избегаем неопределенности inf * 0
        
        if si:
            point1 = dotx + p * q[0]
            point2 = dotx + p * q[1]
            S.append(np.concatenate([point1, point2, [i]]))
    
    return np.array(S)

def ParticularPoints(S, A, b):
    if S.size == 0:
        return np.array([]), 0, np.array([])
    
    PP = []
    V = S[:, :2]
    
    binf = ~((np.abs(V[:, 0]) < np.inf) & (np.abs(V[:, 1]) < np.inf))
    
    if np.any(binf):
        for k in S[binf, 4].astype(int):
            try:
                PP.append((A[k] * b[k]) / np.dot(A[k], A[k]))
            except ZeroDivisionError:
                print(f"Ошибка: Нулевой знаменатель при вычислении PP для строки {k}. Пропускаем.")
                continue
    else:
        PP = V.tolist()
    
    PP = np.array(PP)
    nV = len(PP) if PP.size > 0 else 0
    
    return PP, nV, binf

def Intervals2Path(S):
    if S.size == 0:
        return np.array([])
    
    bs, bp = S[0, :2], S[0, :2]
    P = [bp]
    
    while len(S) > 0:
        distances = np.linalg.norm(S[:, :2] - bs, axis=1)
        index = np.argmin(distances)
        if distances[index] > 1e-8:
            break
        es = S[index, 2:4]
        
        if np.linalg.norm(bs - es) > 1e-8:
            P.append(es)
            if np.linalg.norm(bp - es) < 1e-8:
                break
            bs = es
        S = np.delete(S, index, axis=0)
    
    return np.array(P)

def lineqs(A, b, show=True, title="Solution Set", color='gray',
           bounds=None, alpha=0.5, s=10, size=(15, 15), save=False):
    A = np.asarray(A)
    b = np.asarray(b)
    
    n, m = A.shape
    assert m <= 2, "В матрице A должно быть не более двух столбцов."
    assert b.shape[0] == n, "Размеры матрицы A и вектора b не совпадают."
    
    A, b, cnmty = clear_zero_rows(A, b)
    
    S = BoundaryIntervals(A, b)
    if len(S) == 0:
        return np.array([])
    
    PP, nV, binf = ParticularPoints(S, A, b)
    
    if np.any(binf):
        if bounds is None:
            if PP.size == 0:
                return np.array([])
            PP_min, PP_max = np.min(PP, axis=0), np.max(PP, axis=0)
            center = (PP_min + PP_max) / 2
            rm = max((PP_max - PP_min) / 2)
            A_extended = np.vstack([A, np.eye(2), -np.eye(2)])
            b_extended = np.concatenate([b, center - 5 * rm, -(center + 5 * rm)])
        else:
            A_extended = np.vstack([A, np.eye(2), -np.eye(2)])
            b_extended = np.concatenate([b, [bounds[0][0], bounds[0][1]],
                                         [-bounds[1][0], -bounds[1][1]]])
        
        S = BoundaryIntervals(A_extended, b_extended)
        if len(S) == 0:
            return np.array([])
    
    vertices = Intervals2Path(S)
    vertices = unique(vertices)
    
    if show:
        plt.figure(figsize=size)
        plt.title(title)
        if vertices.size > 0:
            polygon = Polygon(vertices, closed=True, facecolor=color, alpha=alpha)
            plt.gca().add_patch(polygon)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        if save:
            plt.savefig(f"{title}.png")
        plt.show()
    
    return vertices

def IntLinIncR2_fix(A, b, consistency='uni'):
    ortant = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)])
    vertices = []
    n, m = A.shape
    
    assert m <= 2, "В матрице A должно быть не более двух столбцов."
    assert b.shape[0] == n, "Размеры матрицы A и вектора b не совпадают."
    
    def algo(A_local, b_local):
        for ort in range(4):
            tmp_lower = A_local.inf.copy()
            tmp_upper = A_local.sup.copy()
            WorkListA = np.zeros((2 * n + m, m))
            WorkListb = np.zeros(2 * n + m)
    
            for k in range(m):
                if ortant[ort][k] == -1:
                    tmp_lower[:, k] = -tmp_lower[:, k]
                    tmp_upper[:, k] = -tmp_upper[:, k]
                WorkListA[2 * n + k, k] = -ortant[ort][k]
    
            if consistency == 'uni':
                WorkListA[:n] = tmp_lower
                WorkListA[n:2 * n] = -tmp_upper
                WorkListb[:n] = b_local.inf
                WorkListb[n:2 * n] = -b_local.sup
            elif consistency == 'tol':
                WorkListA[:n] = -tmp_lower
                WorkListA[n:2 * n] = tmp_upper
                WorkListb[:n] = -b_local.inf
                WorkListb[n:2 * n] = b_local.sup
            else:
                raise ValueError("Неверно указан тип согласования системы! Используйте 'uni' или 'tol'.")
    
            current_vertices = lineqs(-WorkListA, -WorkListb, show=False)
            if current_vertices.size > 0:
                vertices.append(current_vertices)
            else:
                print(f"Внимание: Не найдены вершины для ортанта {ort}.")
    
    algo(A, b)
    
    if any(len(v) == 0 for v in vertices):
        PP = [v for v in vertices if v.size > 0]
        if len(PP) > 0:
            PP = np.vstack(PP)
            PPmin, PPmax = np.min(PP, axis=0), np.max(PP, axis=0)
            center = (PPmin + PPmax) / 2
            rm = max((PPmax - PPmin) / 2)
            A_extended = Interval(np.vstack([A.inf, A.sup, np.eye(2), -np.eye(2)]))
            b_extended = Interval(np.concatenate([b.inf, b.sup, center - 5 * rm, -(center + 5 * rm)]))
            algo(A_extended, b_extended)
        else:
            print("PP пустой после обработки вершин. Невозможно расширить систему.")
    
    return vertices

if __name__ == "__main__":
    directory = os.path.join(DATA_PATH, '04_10_2024_070_068')
    side = "a"
    side_a_1 = load_data(directory, side)
    
    if side_a_1.size == 0:
        print("Ошибка: Данные не были загружены. Проверьте наличие файлов и их содержимое.")
    else:
        build_plots(side_a_1[3][500], 3, 500)
        plt.show()