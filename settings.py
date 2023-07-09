NEW_PROJECT_ARRIVAL_INTERVAL = [30, 40, 35]

STAGE = [
    'pre-optimization',
    'optimizing',
    'post-optimization'
]

RESOURCES = {
    'business_analysts': 5,
    'designers': 5,
    'programmers': 10,
    'testers': 20,
    'maintenance_people': 5
}

PHASES = {
    'requirements_analysis': {'duration_range': [3, 5], 'resource': 'business_analysts'},
    'design': {'duration_range': [5, 10], 'resource': 'designers'},
    'implementation': {'duration_range': [15, 20], 'resource': 'programmers'},
    'testing': {'duration_range': [5, 10], 'resource': 'testers'},
    'maintenance': {'duration_range': [1, 3], 'resource': 'maintenance_people'}
}

PROJECT_TYPES = {
    'small': {'proportion': 0.7, 'requirements': {'business_analysts': 1, 'designers': 1, 'programmers': 2, 'testers': 2, 'maintenance_people': 1}, 'error_probability': 0.1},
    'medium': {'proportion': 0.25, 'requirements': {'business_analysts': 2, 'designers': 2, 'programmers': 4, 'testers': 6, 'maintenance_people': 2}, 'error_probability': 0.2},
    'large': {'proportion': 0.05, 'requirements': {'business_analysts': 5, 'designers': 5, 'programmers': 10, 'testers': 20, 'maintenance_people': 5}, 'error_probability': 0.3},
}

TOTAL_PROJECTS = 100
