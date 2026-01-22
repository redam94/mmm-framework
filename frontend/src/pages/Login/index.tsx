import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { Card, Title, Text, TextInput, Button } from '@tremor/react';
import { KeyIcon, ExclamationCircleIcon, ChartBarIcon } from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/authStore';
import { checkApiHealth } from '../../api/client';

interface LoginFormData {
  apiKey: string;
}

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, setApiKey, isValidating, validationError } = useAuthStore();
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>();

  // Get the intended destination from location state
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';

  // Check API health on mount
  useEffect(() => {
    checkApiHealth().then((healthy) => {
      setApiStatus(healthy ? 'online' : 'offline');
    });
  }, []);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  const onSubmit = async (data: LoginFormData) => {
    const success = await setApiKey(data.apiKey.trim());
    if (success) {
      navigate(from, { replace: true });
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {/* Logo and title */}
        <div className="text-center">
          <div className="flex justify-center">
            <ChartBarIcon className="h-16 w-16 text-blue-600" />
          </div>
          <Title className="mt-4">MMM Studio</Title>
          <Text className="mt-2">
            Marketing Mix Modeling with Bayesian Inference
          </Text>
        </div>

        {/* Login card */}
        <Card className="mt-8">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            <div>
              <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700">
                API Key
              </label>
              <div className="mt-1 relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <KeyIcon className="h-5 w-5 text-gray-400" />
                </div>
                <TextInput
                  id="apiKey"
                  type="password"
                  placeholder="Enter your API key"
                  className="pl-10"
                  error={!!errors.apiKey || !!validationError}
                  {...register('apiKey', {
                    required: 'API key is required',
                    minLength: {
                      value: 8,
                      message: 'API key must be at least 8 characters',
                    },
                  })}
                />
              </div>
              {errors.apiKey && (
                <p className="mt-2 text-sm text-red-600">{errors.apiKey.message}</p>
              )}
              {validationError && (
                <p className="mt-2 text-sm text-red-600">{validationError}</p>
              )}
            </div>

            <Button
              type="submit"
              className="w-full"
              loading={isValidating}
              disabled={apiStatus === 'offline'}
            >
              {isValidating ? 'Validating...' : 'Sign In'}
            </Button>
          </form>

          {/* API status indicator */}
          <div className="mt-6 pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">API Status</span>
              <div className="flex items-center">
                {apiStatus === 'checking' && (
                  <>
                    <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse mr-2" />
                    <span className="text-yellow-600">Checking...</span>
                  </>
                )}
                {apiStatus === 'online' && (
                  <>
                    <span className="h-2 w-2 rounded-full bg-green-400 mr-2" />
                    <span className="text-green-600">Online</span>
                  </>
                )}
                {apiStatus === 'offline' && (
                  <>
                    <span className="h-2 w-2 rounded-full bg-red-400 mr-2" />
                    <span className="text-red-600">Offline</span>
                  </>
                )}
              </div>
            </div>
          </div>

          {apiStatus === 'offline' && (
            <div className="mt-4 p-3 bg-red-50 rounded-md">
              <div className="flex">
                <ExclamationCircleIcon className="h-5 w-5 text-red-400" />
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">
                    API server is not reachable
                  </h3>
                  <p className="mt-1 text-sm text-red-700">
                    Make sure the backend server is running at http://localhost:8000
                  </p>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Help text */}
        <p className="text-center text-sm text-gray-500">
          Need an API key? Contact your administrator or check the documentation.
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
