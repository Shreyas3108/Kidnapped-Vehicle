/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if(is_initialized)
	{
		return ; 
	}
	// Initialize the particles and set number of particles 
	num_particles = 100;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	//   x, y, theta normal distribution
	normal_distribution <double> dist_x(x,std_x) ;
	normal_distribution <double> dist_y(y , std_y) ;
	normal_distribution <double> dist_theta(theta , std_theta) ; 
	
	// engine 
	default_random_engine gen ; 
	// Add random Gaussian noise to each particle
	for(int i = 0 ; i < num_particles ; ++i) 
	{
		double sample_x, sample_y, sample_theta;
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);
		Particle particle;
		particle.id = i;
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_theta;
		particle.weight = 1.0; //weight 1.0
		particles.push_back(particle);
		weights.push_back(particle.weight); 
	}
	
	is_initialized = true ; 
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_x*delta_t);
	normal_distribution<double> dist_y(0, std_y*delta_t);
	normal_distribution<double> dist_theta(0, std_theta*delta_t);
	//Add measurements to each particle and add random Gaussian noise.
	for (int i = 0; i < num_particles; ++i) {
			Particle &particle = particles[i];  
			if (fabs(yaw_rate) == 0) 
			{
				particle.x += velocity*delta_t*cos(particle.theta) + dist_x(gen);
				particle.y += velocity*delta_t*sin(particle.theta) + dist_y(gen);
				particle.theta= particle.theta + dist_theta(gen);
			}
			else
			{
				particle.x += (velocity/yaw_rate)*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta)) + dist_x(gen);
				particle.y += (velocity/yaw_rate)*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t)) + dist_y(gen);
				particle.theta += yaw_rate*delta_t + dist_theta(gen);
			}
			}



}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  weights.clear();
  for(int i = 0; i<num_particles; i++){

    std::vector<LandmarkObs> predicted;
    double x_p = particles[i].x;
    double y_p = particles[i].y;
    double theta_p = particles[i].theta;
    //Transform car observations to map coordinates supposing that the particle is the car.
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();
    double weight = 1;
    for(int j = 0; j<observations.size(); j++){
      double o_x = observations[j].x;
      double o_y = observations[j].y;
      double o_x_map = o_x * cos(theta_p) - o_y * sin(theta_p) + x_p;
      double o_y_map = o_x * sin(theta_p) + o_y * cos(theta_p) + y_p;
      if(pow(pow(o_x_map-x_p,2)+pow(o_y_map-y_p,2),0.5) > sensor_range) continue;
      particles[i].sense_x.push_back(o_x_map);
      particles[i].sense_y.push_back(o_y_map);
      double min_range = 1000000000;
      int min_k=-1;
      for(int k = 0; k<map_landmarks.landmark_list.size(); k++){

        double l_x = map_landmarks.landmark_list[k].x_f;
        double l_y = map_landmarks.landmark_list[k].y_f;       
        double diff_x = l_x - o_x_map;
        double diff_y = l_y - o_y_map;
        double range = pow(pow(diff_x,2)+pow(diff_y,2),0.5);
        if(range < min_range){
          min_range = range;
          min_k = k;
        }
      }
      double l_x = map_landmarks.landmark_list[min_k].x_f;
      double l_y = map_landmarks.landmark_list[min_k].y_f;
      particles[i].associations.push_back(map_landmarks.landmark_list[min_k].id_i);
      weight = weight * exp(-0.5 * (pow((l_x - o_x_map) / std_landmark[0],2) + pow((l_y - o_y_map) / std_landmark[1],2))) / (2*M_PI*std_landmark[0]*std_landmark[1]);
    } 
    particles[i].weight=weight;
    weights.push_back(weight); 
  }		
  return;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles;
  weights.clear();
  for(int i=0; i < num_particles; i++){
    int chosen = distribution(gen);
    resampled_particles.push_back(particles[chosen]);
    weights.push_back(particles[chosen].weight);
  }
  particles=resampled_particles;
  return;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
