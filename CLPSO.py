import numpy as np #type:ignore

def construct_exemplar(i, Pc_i, n_dim, particles_P_best, particles_fitness):
    """
    Tạo ra vector Exemplar cho hạt thứ i.
    
    i: Index của hạt hiện tại
    Pc_i: Xác suất học tập của hạt i (0.0 đến 1.0)
    n_dim: Số chiều
    particles_P_best: Mảng chứa P_best của tất cả các hạt (Shape: [N, Dim])
    particles_fitness: Mảng chứa giá trị fitness P_best của tất cả các hạt
    """
    N = len(particles_fitness)
    exemplar = np.zeros(n_dim)
    
    # Duyệt qua từng chiều (dimension)
    for d in range(n_dim):
        rand_val = np.random.rand()
        
        if rand_val > Pc_i:
            # Học từ chính mình
            exemplar[d] = particles_P_best[i, d]
        else:
            # Học từ người khác (Tournament Selection)
            # Chọn ngẫu nhiên 2 hạt r1, r2 khác hạt i
            r1 = np.random.randint(0, N)
            r2 = np.random.randint(0, N)
            
            # Đảm bảo không chọn trùng chính mình
            while r1 == i: r1 = np.random.randint(0, N)
            while r2 == i: r2 = np.random.randint(0, N)
            
            # So sánh fitness (Giả sử bài toán Minimize: càng nhỏ càng tốt)
            winner = r1 if particles_fitness[r1] < particles_fitness[r2] else r2
            
            # Lấy giá trị chiều d của người thắng cuộc
            exemplar[d] = particles_P_best[winner, d]
            
    return exemplar

class ComprehensiveLearningPSO:
    def __init__(self, func, n_dim, n_particles, iterations, bounds=(-10, 10)):
        """
        func: Hàm mục tiêu cần tối ưu
        n_dim: Số chiều của bài toán
        n_particles: Số lượng hạt
        iterations: Số vòng lặp
        bounds: Giới hạn không gian tìm kiếm (min, max)
        """
        self.func = func
        self.n_dim = n_dim
        self.n_particles = n_particles
        self.iterations = iterations
        self.bounds = bounds
        
        # Tham số CLPSO
        self.w = 0.7    # Quán tính (Inertia)
        self.c = 1.5    # Hệ số học tập
        
        # Xác suất học tập Pc cho mỗi hạt (có thể khác nhau hoặc giống nhau)
        # Pc càng cao, càng có xu hướng học từ người khác
        self.Pc = np.full(n_particles, 0.5)  # Tất cả hạt có Pc = 0.5
        
        # Tham số refresh exemplar (mỗi m generations không cải thiện thì refresh)
        self.m = 7  # Số thế hệ không cải thiện trước khi refresh
        self.flags = np.zeros(n_particles)  # Đếm số thế hệ không cải thiện
        
        # Khởi tạo vị trí và vận tốc ngẫu nhiên
        self.X = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dim))
        self.V = np.random.uniform(-1, 1, (n_particles, n_dim))
        
        # Khởi tạo P_best (tốt nhất của từng hạt)
        self.P_best_pos = self.X.copy()
        self.P_best_val = np.array([func(x) for x in self.X])
        
        # Khởi tạo G_best (tốt nhất của cả đàn) - để theo dõi
        best_idx = np.argmin(self.P_best_val)
        self.G_best_pos = self.P_best_pos[best_idx].copy()
        self.G_best_val = self.P_best_val[best_idx]
        
        # Khởi tạo exemplar cho mỗi hạt
        self.exemplars = np.zeros((n_particles, n_dim))
        for i in range(n_particles):
            self.exemplars[i] = construct_exemplar(i, self.Pc[i], n_dim, 
                                                   self.P_best_pos, self.P_best_val)

    def optimize(self):
        """
        Chạy thuật toán CLPSO
        """
        for i in range(self.iterations):
            # Tạo số ngẫu nhiên r
            r = np.random.rand(self.n_particles, self.n_dim)
            
            # Kiểm tra và refresh exemplar nếu cần
            for j in range(self.n_particles):
                if self.flags[j] >= self.m:
                    # Refresh exemplar cho hạt j
                    self.exemplars[j] = construct_exemplar(j, self.Pc[j], self.n_dim,
                                                          self.P_best_pos, self.P_best_val)
                    self.flags[j] = 0
            
            # --- CẬP NHẬT VẬN TỐC ---
            # V_new = w*V + c*r*(exemplar - X)
            # Khác với PSO thông thường: không dùng G_best, mà dùng exemplar riêng cho mỗi hạt
            self.V = (self.w * self.V) + \
                     (self.c * r * (self.exemplars - self.X))
            
            # --- CẬP NHẬT VỊ TRÍ ---
            self.X = self.X + self.V
            
            # Giới hạn vị trí trong bounds
            self.X = np.clip(self.X, self.bounds[0], self.bounds[1])
            
            # --- ĐÁNH GIÁ VÀ CẬP NHẬT ---
            # Tính giá trị hàm mục tiêu tại vị trí mới
            current_vals = np.array([self.func(x) for x in self.X])
            
            # Cập nhật P_best và flags
            for j in range(self.n_particles):
                if current_vals[j] < self.P_best_val[j]:
                    # Tìm thấy giải pháp tốt hơn
                    self.P_best_pos[j] = self.X[j].copy()
                    self.P_best_val[j] = current_vals[j]
                    self.flags[j] = 0  # Reset flag
                    # Cập nhật exemplar ngay lập tức khi tìm thấy giải pháp tốt hơn
                    self.exemplars[j] = construct_exemplar(j, self.Pc[j], self.n_dim,
                                                          self.P_best_pos, self.P_best_val)
                else:
                    # Không cải thiện
                    self.flags[j] += 1
            
            # Cập nhật G_best (để theo dõi)
            min_val_idx = np.argmin(self.P_best_val)
            if self.P_best_val[min_val_idx] < self.G_best_val:
                self.G_best_val = self.P_best_val[min_val_idx]
                self.G_best_pos = self.P_best_pos[min_val_idx].copy()
            
            # In kết quả tốt nhất sau mỗi vòng lặp
            print(f"Vòng lặp {i+1}: G_best = {self.G_best_val:.6f}, Vị trí = {self.G_best_pos}")
                
        return self.G_best_pos, self.G_best_val


# Hàm mục tiêu: f(x) = tổng bình phương (Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Thiết lập: 2 chiều (x, y), 30 hạt, 50 vòng lặp
clpso = ComprehensiveLearningPSO(func=sphere_function, n_dim=2, n_particles=30, iterations=50)

print("Bắt đầu tối ưu hóa với CLPSO...")
best_pos, best_val = clpso.optimize()

print("-" * 30)
print(f"Vị trí tối ưu tìm được: {best_pos}")
print(f"Giá trị nhỏ nhất: {best_val}")
