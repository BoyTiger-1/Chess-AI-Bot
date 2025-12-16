import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { readJson } from '@/utils/storage';

const PREFS_KEY = 'aiba.preferences.v1';

const resources = {
  en: {
    translation: {
      nav: {
        home: 'Home',
        market: 'Market Analysis',
        finance: 'Financial Forecasting',
        competitive: 'Competitive Intelligence',
        customer: 'Customer Insights',
        recommendations: 'Recommendation Center',
        settings: 'Settings',
        profile: 'Profile'
      },
      common: {
        search: 'Search',
        save: 'Save',
        cancel: 'Cancel',
        exportPdf: 'Export PDF',
        live: 'Live',
        disconnected: 'Disconnected'
      },
      auth: {
        signIn: 'Sign in',
        email: 'Email',
        password: 'Password',
        role: 'Role',
        signOut: 'Sign out'
      },
      onboarding: {
        title: 'Quick tour',
        body: 'Customize your dashboards, search across modules, and export reports as PDFs.',
        cta: 'Got it'
      }
    }
  },
  es: {
    translation: {
      nav: {
        home: 'Inicio',
        market: 'Análisis de mercado',
        finance: 'Pronóstico financiero',
        competitive: 'Inteligencia competitiva',
        customer: 'Insights de clientes',
        recommendations: 'Centro de recomendaciones',
        settings: 'Configuración',
        profile: 'Perfil'
      },
      common: {
        search: 'Buscar',
        save: 'Guardar',
        cancel: 'Cancelar',
        exportPdf: 'Exportar PDF',
        live: 'En vivo',
        disconnected: 'Desconectado'
      },
      auth: {
        signIn: 'Iniciar sesión',
        email: 'Correo',
        password: 'Contraseña',
        role: 'Rol',
        signOut: 'Cerrar sesión'
      },
      onboarding: {
        title: 'Tour rápido',
        body: 'Personaliza tus tableros, busca en módulos y exporta reportes en PDF.',
        cta: 'Entendido'
      }
    }
  }
} as const;

const initialLang = (() => {
  const prefs = readJson<{ language?: string }>(PREFS_KEY, {});
  return prefs.language === 'es' ? 'es' : 'en';
})();

i18n.use(initReactI18next).init({
  resources,
  lng: initialLang,
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false
  }
});

export default i18n;
