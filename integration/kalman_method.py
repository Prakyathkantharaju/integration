import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise





class AccIntergration:
    def __init__(self, dimension: int = 3, initialization: np.ndarray|None = None, dt: float = 0.01, remove_gravity: int|bool = False) -> None:
        
        self.dt = dt
        self.dimension = dimension
        self.kalmann_filter = KalmanFilter(dim_x=dimension * 3, dim_z=3)
        if initialization is not None:
            self.kalmann_filter.x = initialization
        else:
            self.kalmann_filter.x = np.zeros((dimension*3, 1))
        F = np.eye(dimension * 3)
        F[0:3, 0:3] = self.base_F # For X dimension
        F[3:6, 3:6] = self.base_F # For Y dimension
        F[3:6, 6:9] = self.base_F # For Z dimension

        #TODO: Need to automate this logic
        if dimension == 1:
            H = np.zeros((dimension * 3, 1))
            H[2, 0] = 1
        elif dimension == 2:
            H = np.zeros((dimension * 3, 2))
            H[2, 0] = 1
            H[5, 1] = 1
        elif dimension == 3:
            H = np.zeros((dimension * 3, 3))
            H[2, 0] = 1
            H[5, 1] = 1
            H[8, 2] = 1

        self.kalmann_filter.F = F
        self.kalmann_filter.H = H.T
        self.kalmann_filter.P *= 1000.                       # covariance matrix
        self.kalmann_filter.R = 5                            # state uncertainty
        #TODO: need to streamline this.
        Q = Q_discrete_white_noise(3, 1, .001) # process uncertainty
        Q_full = np.zeros((9, 9))
        Q_full[0:3, 0:3] = Q
        Q_full[3:6, 3:6] = Q
        Q_full[6:9, 6:9] = Q
        self.kalmann_filter.Q = Q_full
        self.position_transform = np.zeros((dimension * 3, dimension))
        self.position_transform[0, 0] = 1
        self.position_transform[3, 1] = 1
        self.position_transform[6, 2] = 1
        self.position_transform = self.position_transform.T

        # removing gravity
        if type(remove_gravity) == bool:
            self.remove_gravity = True
            self.gravity_vector = 3
        else:
            self.remove_gravity = False
            self.gravity_vector = None

    def predict(self, acceleration: np.ndarray) -> None:
        assert acceleration.shape == (self.dimension, 1)
        self.kalmann_filter.predict()
        self.kalmann_filter.update(acceleration)

        return np.dot(self.position_transform, self.kalmann_filter.x)
        



class GyroIntegration:
    def __init__(self, dimension: int = 3, initialization: np.ndarray|None = None, dt: float = 0.01) -> None:
        self.dt = dt
        self.base_F = np.array([[1.,1 * dt],[0.,1.]])
        self.dimension = dimension
        self.kalman_filter = KalmanFilter(dim_x=dimension * 2, dim_z=3)
        if initialization is not None:
            self.kalman_filter.x = initialization
        else:
            self.kalman_filter.x = np.zeros((dimension*2, 1))
        F = np.eye(dimension * 2)
        F[0:2, 0:2] = self.base_F
        F[2:4, 2:4] = self.base_F
        F[4:6, 4:6] = self.base_F

        if dimension == 1:
            H = np.zeros((dimension * 2, 1))
            H[1, 0] = 1
        elif dimension == 2:
            H = np.zeros((dimension * 2, 2))
            H[1, 0] = 1
            H[3, 1] = 1
        elif dimension == 3:
            H = np.zeros((dimension * 2, 3))
            H[1, 0] = 1
            H[3, 1] = 1
            H[5, 2] = 1

        self.kalman_filter.F = F
        self.kalman_filter.H = H.T
        self.kalman_filter.P *= 1000.
        self.kalman_filter.R = 5
        Q = Q_discrete_white_noise(2, 1, .001) 
        Q_full = np.eye(dimension*2)
        Q_full[0:2, 0:2] = Q
        Q_full[2:4, 2:4] = Q
        Q_full[4:6, 4:6] = Q

        self.kalman_filter.Q = Q_full

        self.angle_transform = np.zeros((dimension*2, dimension))
        self.angle_transform[0, 0] = 1
        self.angle_transform[2, 1] = 1
        self.angle_transform[4, 2] = 1
        self.angle_transform = self.angle_transform.T

    def predict(self, gyro: np.ndarray) -> None:
        assert gyro.shape == (self.dimension, 1)
        self.kalman_filter.predict()
        self.kalman_filter.update(gyro)
        return np.dot(self.angle_transform, self.kalman_filter.x)



if __name__ == "__main__":
    AccIntergtation = AccIntergration(3)

    position_store  = []
    for t in range(100):
        # Note that you need to remove the gravity from the system to work.
        position_store.append(AccIntergtation.predict(np.array([[0], [0], [0]]).reshape(3,1)))

    position_store = np.array(position_store)

    plt.plot(position_store[:, 0])
    plt.plot(position_store[:, 1])
    plt.plot(position_store[:, 2])
    plt.show()

    GyroIntergration = GyroIntegration()
    angle_store = []

    for t in range(100):
        angle_store.append(GyroIntergration.predict(np.array([[0], [0], [0]]).reshape(3,1)))

    angle_store = np.array(angle_store)
    plt.plot(angle_store[:, 0])
    plt.plot(angle_store[:, 1])
    plt.plot(angle_store[:, 2])
    plt.show()

