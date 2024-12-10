import numpy as np

INFINITY = np.inf

def get_unique_points(arr, decimals=12):

    rounded = np.round(arr, decimals=decimals)
    unique_arr, indices = np.unique(rounded.view([('', rounded.dtype)]*rounded.shape[1]), return_index=True)
    sorted_indices = np.sort(indices)
    return arr[sorted_indices]

def remove_zero_rows(A, b, dimensions=2):

    A_rounded = np.round(A, decimals=12)
    b_rounded = np.round(b, decimals=12)
    
    condition = (np.sum(np.abs(A_rounded) <= 1e-12, axis=1) == dimensions) & (b_rounded > 0)
    is_consistent = not np.any(condition)
    
    valid_indices = np.where(np.sum(np.abs(A_rounded) <= 1e-12, axis=1) != dimensions)[0]
    return A_rounded[valid_indices], b_rounded[valid_indices], is_consistent

def compute_boundary_intervals(A, b):

    m, n = A.shape
    solutions = []

    for i in range(m):
        q = [-INFINITY, INFINITY]
        feasible = True
        projection = (A[i] * b[i]) / np.dot(A[i], A[i])
        orthogonal = np.array([-A[i, 1], A[i, 0]])

        for k in range(m):
            if k == i:
                continue
            Akx = np.dot(A[k], projection)
            c = np.dot(A[k], orthogonal)

            if c < 0:
                tmp = (b[k] - Akx) / c
                q[1] = min(q[1], tmp)
            elif c > 0:
                tmp = (b[k] - Akx) / c
                q[0] = max(q[0], tmp)
            else:
                if Akx < b[k]:
                    if np.dot(A[k], A[i]) > 0:
                        feasible = False
                        break
                    else:
                        return np.array([])

        if q[0] > q[1]:
            feasible = False

        orthogonal += 1e-301  # Избежание неопределённости inf * 0
        if feasible:
            solutions.append(np.concatenate((projection + orthogonal * q[0], projection + orthogonal * q[1], [i])))

    return np.array(solutions)

def identify_particular_points(S, A, b):

    if S.size == 0:
        return np.array([]), 0, np.array([])
    
    PP = []
    vertices = S[:, :2]
    has_infinite = ~((np.abs(vertices[:, 0]) < INFINITY) & (np.abs(vertices[:, 1]) < INFINITY))
    
    if np.any(has_infinite):
        for k in S[:, 2].astype(int):
            PP.append((A[k] * b[k]) / np.dot(A[k], A[k]))
    else:
        PP = vertices.tolist()
    
    return np.array(PP), np.sum(has_infinite), has_infinite

def convert_intervals_to_path(S):

    if S.size == 0:
        return np.array([])

    path = [S[0, :2]]
    current = S[0, :2]

    while S.size > 0:
        diffs = np.abs(S[:, :2] - current)
        close_indices = np.where(np.all(diffs < 1e-8, axis=1))[0]
        if close_indices.size == 0:
            break
        idx = close_indices[0]
        endpoint = S[idx, 2:4]
        if np.max(np.abs(endpoint - current)) > 1e-8:
            path.append(endpoint)
            current = endpoint
        S = np.delete(S, idx, axis=0)

    return get_unique_points(np.array(path))

def visualize_solution_set(A, b, show=True, title="Solution Set", color='gray', bounds=None, alpha=0.5, point_size=10, figsize=(10, 10), save=False):

    A = np.asarray(A)
    b = np.asarray(b)
    n, m = A.shape
    
    if m != 2:
        raise ValueError("Матрица A должна иметь ровно две колонки.")
    if b.shape[0] != n:
        raise ValueError("Размер вектора b должен соответствовать количеству строк в A.")
    
    A_clean, b_clean, is_consistent = remove_zero_rows(A, b)
    if not is_consistent:
        print("Система несовместна после удаления нулевых строк.")
        return []
    
    S = compute_boundary_intervals(A_clean, b_clean)
    PP, num_infinite, has_infinite = identify_particular_points(S, A_clean, b_clean)
    
    if num_infinite > 0:
        if bounds is None:
            PP_min = np.min(PP, axis=0)
            PP_max = np.max(PP, axis=0)
            center = (PP_min + PP_max) / 2
            margin = max((PP_max - PP_min) / 2) * 5
            bounds = [[center[0] - margin, center[1] - margin], [center[0] + margin, center[1] + margin]]
        
        A_extended = np.vstack([A_clean, np.eye(2), -np.eye(2)])
        b_extended = np.concatenate([b_clean, [bounds[0][0], bounds[0][1]], [-bounds[1][0], -bounds[1][1]]])
        S = compute_boundary_intervals(A_extended, b_extended)
        PP, num_infinite, has_infinite = identify_particular_points(S, A_extended, b_extended)
    
    path = convert_intervals_to_path(S)
    vertices = get_unique_points(path)
    
    if show:
        plt.figure(figsize=figsize)
        plt.title(title)
        if vertices.size > 0:
            polygon = Polygon(vertices, closed=True, fill=True, color=color, alpha=alpha)
            plt.gca().add_patch(polygon)
        plt.xlabel("B0")
        plt.ylabel("B1")
        plt.grid(True)
        if save:
            plt.savefig(f"{title}.png")
        plt.show()
    
    return vertices.tolist()

def IntLinIncR2(A, b, consistency='uni'):

    ortants = [ (1, 1), (-1, 1), (-1, -1), (1, -1) ]
    vertices = []
    n, m = A.shape
    
    def process_ortant(bounds):
        for ort in ortants:
            temp_A = A.copy()
            temp_A[:, 0] *= ort[0]
            temp_A[:, 1] *= ort[1]
            WorkListA = np.zeros((2 * n + m, m))
            WorkListb = np.zeros(2 * n + m)
            
            # Заполнение WorkListA и WorkListb в зависимости от типа согласования
            if consistency == 'uni':
                WorkListA[:n, :] = temp_A
                WorkListA[n:2*n, :] = -temp_A
                WorkListb[:n] = b
                WorkListb[n:2*n] = -b
            elif consistency == 'tol':
                WorkListA[:n, :] = -temp_A
                WorkListA[n:2*n, :] = temp_A
                WorkListb[:n] = -b
                WorkListb[n:2*n] = b
            else:
                raise ValueError("Некорректный тип согласования. Используйте 'uni' или 'tol'.")
            
            # Добавление дополнительных ограничений для ортанта
            WorkListA[2*n + m - m:m] = -ort
            WorkListb[2*n + m - m:m] = 0
            
            # Визуализация или дополнительная обработка
            vertices.extend(visualize_solution_set(-WorkListA, -WorkListb, show=False))
    
    process_ortant(None)
    
    # Обработка неограниченных множеств
    if len(vertices) == 0:
        return []
    
    unique_vertices = get_unique_points(np.array(vertices))
    return unique_vertices.tolist()