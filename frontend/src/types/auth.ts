export type UserRole = 'executive' | 'analyst' | 'stakeholder' | 'admin';

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: UserRole;
}
