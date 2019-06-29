/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>
#include <array>
#include <map>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  // num_particles = 1000;  // TODO: Set the number of particles
  particles.reserve(num_particles);
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  for (int i=0; i<num_particles; ++i)
  {
    Particle temp;
    temp.id = i;
    temp.x = dist_x(gen);
    temp.y = dist_y(gen);
    temp.theta = dist_theta(gen);
    temp.weight = 1.0;
    particles.push_back(temp);
  }
  is_initialized = true;
  std::cout<<"new "<<particles.size()<<" particals created"<<std::endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);
  if (yaw_rate > 0.001)
  {
    for (int i=0; i<num_particles; ++i)
    {
      double new_theta = yaw_rate*delta_t + particles[i].theta;
      particles[i].x += (velocity/yaw_rate*(sin(new_theta) - 
                        sin(particles[i].theta)) + dist_x(gen));
      particles[i].y += (velocity/yaw_rate*(cos(particles[i].theta) - 
                        cos(new_theta)) + dist_y(gen));
      particles[i].theta = new_theta + dist_theta(gen);
    }
  }
  else
  {
    for (int i=0; i<num_particles; ++i)
    {
      particles[i].x += (velocity*delta_t*cos(particles[i].theta) + dist_x(gen));
      particles[i].y += (velocity*delta_t*sin(particles[i].theta) + dist_x(gen));
      particles[i].theta += (yaw_rate*delta_t + dist_theta(gen));
    }
  }
}

void ParticleFilter::dataAssociation(const Map &map_landmarks, 
                                     vector<LandmarkObs>& observations) {
// void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
//                                      vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (auto &obsv_pnt_map : observations)
  {
    double dist_sqr = std::numeric_limits<double>::max();
    for (auto landmark : map_landmarks.landmark_list)
    {
      double dist_x = obsv_pnt_map.x - landmark.x_f;
      double dist_y = obsv_pnt_map.y - landmark.y_f;
      if (dist_x*dist_x + dist_y*dist_y < dist_sqr)
      {
        dist_sqr = dist_x*dist_x + dist_y*dist_y;
        obsv_pnt_map.id = landmark.id_i;
      }
    }
    // std::cout<<"dist_sqr "<<dist_sqr<<std::endl;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i=0; i<num_particles; ++i)
  {
    vector<LandmarkObs> obsv_pnt_map(0);
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();  
    for (auto obsv_pnt : observations)
    {
      double x_map_temp = particles[i].x + (cos(particles[i].theta)*obsv_pnt.x) 
                        - (sin(particles[i].theta)*obsv_pnt.y);
      double y_map_temp = particles[i].y + (sin(particles[i].theta)*obsv_pnt.x) 
                        + (cos(particles[i].theta)*obsv_pnt.y);  
      particles[i].sense_x.emplace_back(x_map_temp);
      particles[i].sense_y.emplace_back(y_map_temp);
      // std::cout<<"sense_x size "<<particles[i].sense_x.size()<<", sense_y size "<<particles[i].sense_y.size()<<std::endl;
      obsv_pnt_map.emplace_back(x_map_temp, y_map_temp);
    }
    dataAssociation(map_landmarks, obsv_pnt_map);

    particles[i].associations.clear();
    for (auto ass_pnt_map : obsv_pnt_map)
    {
      // std::cout<<"ass_pnt_map.id "<<ass_pnt_map.id<<", x "<<ass_pnt_map.x <<std::endl;
      double weight = 1.0;
      for (auto landmark : map_landmarks.landmark_list)
      {
        if (ass_pnt_map.id == landmark.id_i)
        {
          weight *= multiv_prob(landmark.x_f, landmark.y_f, 
                                  ass_pnt_map.x, ass_pnt_map.y, std_landmark);
          // std::cout<<landmark.x_f<<", "<<landmark.y_f<<", "<<ass_pnt_map.x<<", "<<ass_pnt_map.y<<std::endl;
        }
      }
      particles[i].weight = weight;
      particles[i].associations.push_back(ass_pnt_map.id);
      // std::cout<<"associations size "<< particles[i].associations.size()<<std::endl;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> new_particles(0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::array<double, num_particles> weights;
  for (int i=0; i<num_particles; ++i)
  {
    weights[i] = particles[i].weight;
  }
  std::discrete_distribution<> d{std::begin(weights), std::end(weights)};
  // std::map<int, int> m;
  for (int i=0; i<num_particles; ++i)
  {
    auto temp_particle = particles[d(gen)];
    // ++m[temp_particle.id];
    temp_particle.id = i;
    new_particles.push_back(temp_particle);
  }
  // for(auto p : m) 
  // {
  //   if (p.second >2)
  //       std::cout << p.first << " generated " << p.second << " times\n";
  // }
  particles = new_particles;

  double sum_weight = 0.0;
  for (int i=0; i<num_particles; ++i)
  {
    sum_weight += particles[i].weight;
  }
  for (int i=0; i<num_particles; ++i)
  {
    particles[i].weight /= sum_weight;
  }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}