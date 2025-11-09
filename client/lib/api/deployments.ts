/**
 * Azure ML Deployments API Client
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Deployment {
  id: string;
  user_id: string;
  model_id: string;
  deployment_name: string;
  endpoint_name: string;
  azure_model_name: string;
  azure_model_version?: string;
  scoring_uri?: string;
  swagger_uri?: string;
  instance_type: string;
  instance_count: number;
  status: 'deploying' | 'active' | 'failed' | 'deleted';
  error_message?: string;
  description?: string;
  deployment_config?: Record<string, any>;
  created_at: string;
  deployed_at?: string;
  deleted_at?: string;
  trained_models?: {
    model_name: string;
    algorithm: string;
    model_type: string;
  };
}

export interface DeploymentStats {
  total_deployments: number;
  active_deployments: number;
  deploying_count: number;
  failed_deployments: number;
  last_deployment_date?: string;
}

export interface DeployModelRequest {
  model_id: string;
  endpoint_name?: string;
  instance_type?: string;
  instance_count?: number;
  description?: string;
}

export interface DeployModelResponse {
  deployment_id: string;
  status: string;
  message: string;
  endpoint_name?: string;
  scoring_uri?: string;
}

/**
 * Deploy a model to Azure ML
 */
export async function deployModel(
  request: DeployModelRequest,
  userId: string
): Promise<DeployModelResponse> {
  const response = await fetch(`${API_BASE}/deployments/deploy?user_id=${userId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to deploy model');
  }

  return response.json();
}

/**
 * List all deployments for a user
 */
export async function listDeployments(
  userId: string,
  status?: 'deploying' | 'active' | 'failed'
): Promise<{ deployments: Deployment[]; total: number }> {
  const url = new URL(`${API_BASE}/deployments/list`);
  url.searchParams.append('user_id', userId);
  if (status) {
    url.searchParams.append('status', status);
  }

  const response = await fetch(url.toString());

  if (!response.ok) {
    throw new Error('Failed to list deployments');
  }

  return response.json();
}

/**
 * Get deployment statistics for a user
 */
export async function getDeploymentStats(
  userId: string
): Promise<DeploymentStats> {
  const response = await fetch(
    `${API_BASE}/deployments/stats?user_id=${userId}`
  );

  if (!response.ok) {
    throw new Error('Failed to get deployment stats');
  }

  return response.json();
}

/**
 * Delete a deployment
 */
export async function deleteDeployment(
  deploymentId: string,
  userId: string
): Promise<{ message: string }> {
  const response = await fetch(
    `${API_BASE}/deployments/${deploymentId}?user_id=${userId}`,
    {
      method: 'DELETE',
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete deployment');
  }

  return response.json();
}
