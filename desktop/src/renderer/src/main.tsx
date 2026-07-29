import React from 'react'
import ReactDOM from 'react-dom/client'
import { LazyMotion, domMax } from 'motion/react'
import App from './App'
import './styles.css'

// LazyMotion + `m.` components keep the motion bundle lazy/small (DESIGN.md
// §5.1). domMax (not domAnimation) because the sliding nav highlight uses a
// shared-element `layoutId` transition, which needs the layout feature set.
// `strict` throws if a full `motion.` component sneaks in.
ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <LazyMotion features={domMax} strict>
      <App />
    </LazyMotion>
  </React.StrictMode>
)
