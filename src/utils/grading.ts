export const calculateGrade = (defects: number): 'A' | 'B' | 'C' => {
  if (defects === 0) return 'A';
  if (defects <= 3) return 'B';
  return 'C';
};

export const getGradeColor = (grade: 'A' | 'B' | 'C'): string => {
  switch (grade) {
    case 'A': return 'text-green-600 bg-green-50';
    case 'B': return 'text-yellow-600 bg-yellow-50';
    case 'C': return 'text-red-600 bg-red-50';
  }
};

export const getRipenessColor = (ripeness: 'ripe' | 'unripe'): string => {
  return ripeness === 'ripe' 
    ? 'text-green-600 bg-green-50' 
    : 'text-orange-600 bg-orange-50';
};