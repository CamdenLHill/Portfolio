% Camden Hill
% Vibrations 3DOF Spring-Mass-Damper using Rayleigh Damping
format longg

syms w t

damp = [0.03; 0.05; 0.02];
q0 = [0.5; 0.75; 0];
q0_ = [0; 0; 0];
M = [4 0 0; 0 2 0; 0 0 6];
K = [4, -1, 0; -1, 3, -2; 0, -2, 2];
disp("M =")
disp(M)
disp("K =")
disp(K)
A = K - ((w)*M);
Adet = det(A);
disp("Characteristic Equation:")
disp(Adet)
% Calculate the eigenvalues and eigenvectors of matrix A
eigenValues = vpasolve(Adet == 0,w);
eigenValues_real = zeros(1,height(M));
for i = 1:height(M)
    eigenValues_real(i) = eigenValues(i);
end
eigenValues = eigenValues_real;
eigenVectors = zeros(height(M),height(M));
for i = 1:height(M)
    eigenVectors(:,i) = null(double(subs(A, w, eigenValues(i)))); % Calculate eigenvectors
end
normalized_check = [norm(eigenVectors(:,1)), norm(eigenVectors(:,2)), norm(eigenVectors(:,3))];
% Display the eigenvalues and eigenvectors
disp('Eigenvalues:');
disp(eigenValues);
disp("Check of Eigenvalues:")
disp([double(subs(Adet, w, eigenValues(1))), double(subs(Adet, w, eigenValues(2))), double(subs(Adet, w, eigenValues(3)))])
disp('Eigenvectors:');
disp(eigenVectors);
disp("Check if Eigenvectors are normalized:")
disp(normalized_check)

M_diag = eigenVectors.'*M*eigenVectors;
K_diag = eigenVectors.'*K*eigenVectors;
disp("Check if Eigenvectors diagonalize Inertial Matrix:")
disp(M_diag)
disp("Check if Eigenvectors diagonalize Stiffness Matrix:")
disp(K_diag)
disp("Check if Individual Eigenvectors diagonalize:")
disp([eigenVectors(:,1).'*M*eigenVectors(:,1),eigenVectors(:,2).'*M*eigenVectors(:,2)])
disp([eigenVectors(:,1).'*K*eigenVectors(:,1),eigenVectors(:,2).'*K*eigenVectors(:,2)])

a_b = ([1/eigenValues(1), eigenValues(1); 1/eigenValues(3), eigenValues(3)]^-1) * [2*damp(1); 2*damp(3)];
disp("alpha and beta =")
disp(a_b)
new_damp_diag = (eigenVectors.'*a_b(1)*M*eigenVectors) + (eigenVectors.'*a_b(2)*K*eigenVectors);
disp("Diagonalized Rayleigh Damping Matrix =")
disp(new_damp_diag)
C = M_diag\new_damp_diag; % M_diag\new_damp_diag;
disp("Damping Matrix =")
disp(C)
Omegasq = M_diag\K_diag;
disp("Modal Diagonal Matrix =")
disp(Omegasq)

eta0 = eigenVectors\q0; % inv(eigenVectors) * q0; eigenVectors\q0;
eta_0 = eigenVectors\q0_; % inv(eigenVectors) * q0_; eigenVectors\q0_;
disp("Eta0 or Initial Modal Displacements =")
disp(eta0)
disp("Eta_0 or Initial Modal Velocity =")
disp(eta_0)

const_sol_vect = [eta0.';eta_0.'];
const_sol_vect = const_sol_vect(:)'.';
eta_subs = zeros(2*height(M),2*height(M));
for i = 1:height(M)
    val1 = -0.5*C(i,i);
    val2 = 0.5*sqrt((C(i,i).^2) - (4*(Omegasq(i,i))));
    eta_subs((2*(i-1))+1:(2*(i-1))+2,(2*(i-1))+1:(2*(i-1))+2) = [1, 1; val1 - val2, val1 + val2];
end
disp("t = 0 substitutions for Eta =")
disp(eta_subs)
eta_consts = eta_subs\const_sol_vect;
disp("Constants in the Modal Coordinate System =")
disp(eta_consts)
eta_eqs = sym('eta_eqs',[height(M),1]);
for i = 1:height(M)
    val1 = -0.5*C(i,i);
    val2 = 0.5*sqrt((C(i,i).^2) - (4*(Omegasq(i,i))));
    eta_eqs(i) = (eta_consts((2*(i-1))+1)*exp((val1-val2)*t)) + (eta_consts((2*(i-1))+2)*exp((val1+val2)*t));
end
disp("Modal EOM Equations =")
disp(eta_eqs)
disp(vpa(eta_eqs,4))
disp("Eta Verification Check =")
disp(vpa(eta_eqs(1),4))
disp(vpa(diff(eta_eqs(1),t),4))
disp(vpa(diff(diff(eta_eqs(1),t),t),4))
disp(double(subs(diff(diff(eta_eqs(1),t),t) + (C(1,1)*diff(eta_eqs(1),t)) + (Omegasq(1,1)*eta_eqs(1)),t,0)))
disp(double(subs(diff(diff(eta_eqs(2),t),t) + (C(2,2)*diff(eta_eqs(2),t)) + (Omegasq(2,2)*eta_eqs(2)),t,0)))
disp(double(subs(diff(diff(eta_eqs(3),t),t) + (C(3,3)*diff(eta_eqs(3),t)) + (Omegasq(3,3)*eta_eqs(3)),t,0)))

q_eqs = eigenVectors*eta_eqs;
disp("q Equations =")
disp(q_eqs)
disp(vpa(q_eqs,4))
disp(double(real(subs(q_eqs(1),t,0))))

tvals = linspace(0,900,4001);
q1vals = double(subs(q_eqs(1),t,tvals));
q2vals = double(subs(q_eqs(2),t,tvals));
q3vals = double(subs(q_eqs(3),t,tvals));
plot(tvals,q1vals,tvals,q2vals,tvals,q3vals);

% % Missing last mode only
% Mode_num_analyzed = height(M) - 2;
% tvals = linspace(0,10,201).';
% q_eqs_mod = sym('q_eqs_mod',[height(M),1]);
% for i = 1:height(M)
%     q_eqs_mod(i) = 0;
% end
% for i = 1:height(M)
%     Amat = zeros(201,2*Mode_num_analyzed);
%     bvect = double(subs(q_eqs(i),t,tvals));
%     for j = 1:Mode_num_analyzed
%         Amat(:,(2*(j-1))+1:(2*(j-1))+2) = [cos(eigenValues(j)*tvals),sin(eigenValues(j)*tvals)];
%     end
%     Amat = double(Amat);
%     lstqrs_consts = lsqr(Amat,bvect);
%     disp("Lstqrs Fit Constants =")
%     disp(lstqrs_consts)
%     for j = 1:Mode_num_analyzed
%         q_eqs_mod(i) = q_eqs_mod(i) + (lstqrs_consts((2*(j-1))+1)*cos(eigenValues(j)*t)) + (lstqrs_consts((2*(j-1))+2)*sin(eigenValues(j)*t));
%     end
% end
% disp("q Equations Modified by Least-Squares =")
% disp(q_eqs_mod)
% disp(vpa(q_eqs_mod,2))

% Mode_num_analyzed = height(M) - 1;
% IC_mat = horzcat(q0,q0_);
% q0_mod = eigenVectors(:, 1:Mode_num_analyzed)*q_eqs_consts(1:Mode_num_analyzed,:);
% disp("q Components Modified =")
% disp(q0_mod)
% q0_mat = double(subs(q0_mod,t,0));
% disp("q0 Components Modified =")
% disp(q0_mat)
% q0bar = sym('q0bar',[height(M),1]);
% for j = 1:height(M)
%     q0bar(j,1) = q0_mat(j,1) + q0_mat(j,2);
% end
% disp("q0_bar Equation =")
% disp(double(q0bar))
% 
% q0_mod = eigenVectors(:, 1:Mode_num_analyzed)*diff(q_eqs_consts(1:Mode_num_analyzed,:),t);
% disp("q' Components Modified =")
% disp(q0_mod)
% q0_mat = double(subs(q0_mod,t,0));
% disp("q0' Components Modified =")
% disp(q0_mat)
% q0_bar = sym('q0bar',[height(M),1]);
% for j = 1:height(M)
%     q0_bar(j,1) = q0_mat(j,1) + q0_mat(j,2);
% end
% disp("q0_bar' Equation =")
% disp(double(q0_bar))
% 
% q0_bar_combined_mat = horzcat(q0bar,q0_bar);
% disp("q_bar Matrix =")
% disp(q0_bar_combined_mat)
% weight_matrix = ((q0_bar_combined_mat.'*q0_bar_combined_mat)^-1)*q0_bar_combined_mat.'*IC_mat;
% disp("Weight Matrix =")
% disp(weight_matrix)



clear()