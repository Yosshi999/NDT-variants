#include <iostream>

#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/Lie/SE.h>
#include <Eigen/Dense>
using namespace mrpt;
using namespace mrpt::poses;

int main() {
	CPose3D q(1, 2, 3, 0.0_deg, 0.0_deg, 0.0_deg);
	Eigen::VectorXd vec = Lie::SE<3>::log(q).asEigen();
	std::cout << q << " SE(3)::log => " << vec.transpose() << std::endl;
	return 0;
}
