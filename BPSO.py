import numpy as np #type:ignore

class BinaryPSO:
    def __init__(self, func, n_dim, n_particles, iterations):
        self.func = func
        self.n_dim = n_dim
        self.n_particles = n_particles
        self.iterations = iterations
        
        # Tham số
        self.w = 0.7        # Quán tính
        self.c1 = 1.4       # Cá nhân
        self.c2 = 1.4       # Xã hội
        self.V_max = 6.0    # Kẹp vận tốc (Rất quan trọng trong BPSO)

        # Khởi tạo
        # Vị trí ngẫu nhiên 0 hoặc 1
        self.X = np.random.randint(2, size=(n_particles, n_dim))
        
        # Vận tốc thực
        self.V = np.random.uniform(-self.V_max, self.V_max, (n_particles, n_dim))
        
        # P_best & G_best
        self.P_best_pos = self.X.copy()
        self.P_best_val = np.array([self.func(x) for x in self.X])
        
        best_idx = np.argmin(self.P_best_val)
        self.G_best_pos = self.P_best_pos[best_idx].copy()
        self.G_best_val = self.P_best_val[best_idx]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def optimize(self):
        for i in range(self.iterations):
            r1 = np.random.rand(self.n_particles, self.n_dim)
            r2 = np.random.rand(self.n_particles, self.n_dim)
            
            # --- 1. CẬP NHẬT VẬN TỐC ---
            self.V = (self.w * self.V) + \
                     (self.c1 * r1 * (self.P_best_pos - self.X)) + \
                     (self.c2 * r2 * (self.G_best_pos - self.X))
            
            # Kẹp vận tốc để tránh bão hòa Sigmoid
            self.V = np.clip(self.V, -self.V_max, self.V_max)
            
            # --- 2. HÀM CHUYỂN ĐỔI (SIGMOID) ---
            prob_is_one = self.sigmoid(self.V)
            
            # --- 3. CẬP NHẬT VỊ TRÍ (LẬT BIT) ---
            random_threshold = np.random.rand(self.n_particles, self.n_dim)
            # Nếu xác suất > random -> 1, ngược lại -> 0
            self.X = (prob_is_one > random_threshold).astype(int)
            
            # --- 4. ĐÁNH GIÁ ---
            current_vals = np.array([self.func(x) for x in self.X])
            
            # Update P_best
            better_mask = current_vals < self.P_best_val
            self.P_best_pos[better_mask] = self.X[better_mask]
            self.P_best_val[better_mask] = current_vals[better_mask]
            
            # Update G_best
            min_val_idx = np.argmin(self.P_best_val)
            if self.P_best_val[min_val_idx] < self.G_best_val:
                self.G_best_val = self.P_best_val[min_val_idx]
                self.G_best_pos = self.P_best_pos[min_val_idx].copy()
            
            # In kết quả tốt nhất sau mỗi vòng lặp
            print(f"Vòng lặp {i+1}: G_best = {self.G_best_val:.6f}, Vị trí = {self.G_best_pos}")
                
        return self.G_best_pos, self.G_best_val

# --- VÍ DỤ ỨNG DỤNG GIẢ LẬP ---
# Bài toán: Chọn tập con các kênh truyền (Bits 1) sao cho tổng giá trị là lớn nhất
# Nhưng không vượt quá tải trọng (đây là dạng bài toán Knapsack - Cái túi)

# Giả sử ta có 20 kênh, mỗi kênh có "Lợi ích" (Throughput) và "Chi phí" (Power)
np.random.seed(42)
num_channels = 20
throughput = np.random.randint(10, 100, num_channels) # Lợi ích
power_cost = np.random.randint(5, 20, num_channels)   # Chi phí
max_power = 100                                       # Giới hạn công suất

def objective_function(solution_vector):
    # solution_vector là mảng [0, 1, 0, 1...]
    
    total_throughput = np.sum(solution_vector * throughput)
    total_power = np.sum(solution_vector * power_cost)
    
    # Ràng buộc: Nếu tổng công suất vượt quá max_power, phạt thật nặng
    if total_power > max_power:
        return 10000 # Giá trị phạt (Penalty)
    
    # Vì PSO tìm min, mà ta muốn max throughput, nên trả về giá trị âm
    return -total_throughput

# Chạy BPSO
bpso = BinaryPSO(func=objective_function, n_dim=num_channels, n_particles=30, iterations=100)
best_solution, best_score = bpso.optimize()

print(f"Giải pháp chọn kênh tối ưu: {best_solution}")
print(f"Tổng Throughput đạt được: {-best_score}")
print(f"Tổng Power tiêu thụ: {np.sum(best_solution * power_cost)} / {max_power}")